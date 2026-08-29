#include "nn_tensorrt.h"

#include "cppshogi.h"
#include "unpack.h"

class Logger : public nvinfer1::ILogger
{
	const char* error_type(Severity severity)
	{
		switch (severity)
		{
		case Severity::kINTERNAL_ERROR: return "[F] ";
		case Severity::kERROR: return "[E] ";
		case Severity::kWARNING: return "[W] ";
		case Severity::kINFO: return "[I] ";
		case Severity::kVERBOSE: return "[V] ";
		default: assert(0); return "";
		}
	}
	void log(Severity severity, const char* msg) noexcept
	{
		if (severity == Severity::kINTERNAL_ERROR) {
			std::cerr << error_type(severity) << msg << std::endl;
		}
	}
} gLogger;

constexpr long long int operator"" _MiB(long long unsigned int val)
{
	return val * (1 << 20);
}

struct NNTensorRT::InferenceSlot {
	packed_features1_t* p1_dev = nullptr;
	packed_features2_t* p2_dev = nullptr;
	features1_t* x1_dev = nullptr;
	features2_t* x2_dev = nullptr;
	DType* y1_dev = nullptr;
	DType* y2_dev = nullptr;
	cudaStream_t stream = nullptr;
	InferUniquePtr<nvinfer1::IExecutionContext> context;
};

NNTensorRT::NNTensorRT(const char* filename, const int gpu_id, const int max_batch_size, const int profile_count) :
	gpu_id(gpu_id),
	max_batch_size(max_batch_size),
	profile_count(profile_count > 0 ? profile_count : 1)
{
	load_model(filename);
}

NNTensorRT::~NNTensorRT()
{
	for (auto& slot : slots)
	{
		if (slot->stream)
		{
			checkCudaErrors(cudaStreamSynchronize(slot->stream));
			checkCudaErrors(cudaStreamDestroy(slot->stream));
		}
		slot->context.reset();
		checkCudaErrors(cudaFree(slot->p1_dev));
		checkCudaErrors(cudaFree(slot->p2_dev));
		checkCudaErrors(cudaFree(slot->x1_dev));
		checkCudaErrors(cudaFree(slot->x2_dev));
		checkCudaErrors(cudaFree(slot->y1_dev));
		checkCudaErrors(cudaFree(slot->y2_dev));
	}
}

void NNTensorRT::build(const std::string& onnx_filename)
{
	auto builder = InferUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
	if (!builder)
	{
		throw std::runtime_error("createInferBuilder");
	}

	// TensorRT 10以降は暗黙バッチが廃止され、常に明示バッチ（旧kEXPLICIT_BATCH相当）で動作するため
	// フラグ指定は不要になった。
	auto network = InferUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
	if (!network)
	{
		throw std::runtime_error("createNetworkV2");
	}

	auto config = InferUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
	if (!config)
	{
		throw std::runtime_error("createBuilderConfig");
	}

	auto parser = InferUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
	if (!parser)
	{
		throw std::runtime_error("createParser");
	}

	auto parsed = parser->parseFromFile(onnx_filename.c_str(), (int)nvinfer1::ILogger::Severity::kWARNING);
	if (!parsed)
	{
		throw std::runtime_error("parseFromFile");
	}

	config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 64_MiB);

	// TensorRT 11ではIInt8Calibrator系のレガシーキャリブレーションAPIが削除されており、
	// INT8を使う場合は明示的な量子化（QDQノードを含むONNXモデル）が前提となる。
	// FP16/INT8の指定もITensor::setType()ではなくONNXモデル自体の型で決まる仕様に変わったため、
	// ここでは常にONNXモデルに記録された型（通常FP32）のままビルドする。

	assert(network->getNbInputs() == 2);
	nvinfer1::Dims inputDims[] = { network->getInput(0)->getDimensions(), network->getInput(1)->getDimensions() };
	assert(inputDims[0].nbDims == 4);
	assert(inputDims[1].nbDims == 4);

	assert(network->getNbOutputs() == 2);

	const auto dims1 = inputDims[0].d;
	const auto dims2 = inputDims[1].d;
	for (int i = 0; i < profile_count; ++i)
	{
		auto profile = builder->createOptimizationProfile();
		profile->setDimensions("input1", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, dims1[1], dims1[2], dims1[3]));
		profile->setDimensions("input1", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(max_batch_size, dims1[1], dims1[2], dims1[3]));
		profile->setDimensions("input1", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(max_batch_size, dims1[1], dims1[2], dims1[3]));
		profile->setDimensions("input2", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, dims2[1], dims2[2], dims2[3]));
		profile->setDimensions("input2", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(max_batch_size, dims2[1], dims2[2], dims2[3]));
		profile->setDimensions("input2", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(max_batch_size, dims2[1], dims2[2], dims2[3]));
		config->addOptimizationProfile(profile);
	}

	auto serializedEngine = InferUniquePtr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
	if (!serializedEngine)
	{
		throw std::runtime_error("buildSerializedNetwork");
	}
	runtime = InferUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
	engine.reset(runtime->deserializeCudaEngine(serializedEngine->data(), serializedEngine->size()));
	if (!engine)
	{
		throw std::runtime_error("deserializeCudaEngine");
	}
	// 一旦シリアライズ化されたエンジンはデシリアライズを行った上で捨てているが、
	// この後またすぐにファイル書き出し用にシリアライズを行っているので、手順改善の余地あり。
	// // auto serializedEngine = InferUniquePtr<nvinfer1::IHostMemory>(engine->serialize());
}

void NNTensorRT::load_model(const char* filename)
{
	std::string serialized_filename = std::string(filename) + "." + std::to_string(gpu_id) + "." + std::to_string(max_batch_size)
		+ ".profiles" + std::to_string(profile_count)
#ifdef FP16
		+ ".fp16"
#endif
		+ ".serialized";
	std::ifstream seriarizedFile(serialized_filename, std::ios::binary);
	if (seriarizedFile.is_open())
	{
		// deserializing a model
		seriarizedFile.seekg(0, std::ios_base::end);
		const size_t modelSize = seriarizedFile.tellg();
		seriarizedFile.seekg(0, std::ios_base::beg);
		std::unique_ptr<char[]> blob(new char[modelSize]);
		seriarizedFile.read(blob.get(), modelSize);
		runtime = InferUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
		engine = InferUniquePtr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(blob.get(), modelSize));
		if (!engine)
		{
			throw std::runtime_error("deserializeCudaEngine");
		}
	}
	else
	{

		// build
		build(filename);

		// serializing a model
		auto serializedEngine = InferUniquePtr<nvinfer1::IHostMemory>(engine->serialize());
		if (!serializedEngine)
		{
			throw std::runtime_error("Engine serialization failed");
		}
		std::ofstream engineFile(serialized_filename, std::ios::binary);
		if (!engineFile)
		{
			throw std::runtime_error("Cannot open engine file");
		}
		engineFile.write(static_cast<char*>(serializedEngine->data()), serializedEngine->size());
		if (engineFile.fail())
		{
			throw std::runtime_error("Cannot open engine file");
		}
	}

	inputDims1 = engine->getTensorShape("input1");
	inputDims2 = engine->getTensorShape("input2");
}

int NNTensorRT::slot_capacity() const
{
	return profile_count;
}

std::unique_ptr<NNTensorRT::InferenceSlot> NNTensorRT::create_slot(const int profile_index)
{
	if (profile_index < 0 || profile_index >= engine->getNbOptimizationProfiles())
	{
		throw std::out_of_range("NNTensorRT profile_index");
	}

	auto slot = std::unique_ptr<InferenceSlot>(new InferenceSlot());
	checkCudaErrors(cudaMalloc((void**)&slot->p1_dev, sizeof(packed_features1_t) * max_batch_size));
	checkCudaErrors(cudaMalloc((void**)&slot->p2_dev, sizeof(packed_features2_t) * max_batch_size));
	checkCudaErrors(cudaMalloc((void**)&slot->x1_dev, sizeof(features1_t) * max_batch_size));
	checkCudaErrors(cudaMalloc((void**)&slot->x2_dev, sizeof(features2_t) * max_batch_size));
	checkCudaErrors(cudaMalloc((void**)&slot->y1_dev, MAX_MOVE_LABEL_NUM * (size_t)SquareNum * max_batch_size * sizeof(DType)));
	checkCudaErrors(cudaMalloc((void**)&slot->y2_dev, max_batch_size * sizeof(DType)));
	checkCudaErrors(cudaStreamCreateWithFlags(&slot->stream, cudaStreamNonBlocking));
	slot->context = InferUniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
	if (!slot->context)
	{
		throw std::runtime_error("createExecutionContext");
	}
	if (!slot->context->setOptimizationProfileAsync(profile_index, slot->stream))
	{
		throw std::runtime_error("setOptimizationProfileAsync");
	}
	// スロットのデバイスバッファはスロット生存期間中ずっと同じアドレスを使い回すため、
	// テンソルアドレスの登録は生成時に一度だけ行えばよい（バッチサイズが変わっても再設定は不要）。
	if (!slot->context->setTensorAddress("input1", slot->x1_dev) ||
		!slot->context->setTensorAddress("input2", slot->x2_dev) ||
		!slot->context->setTensorAddress("output_policy", slot->y1_dev) ||
		!slot->context->setTensorAddress("output_value", slot->y2_dev))
	{
		throw std::runtime_error("setTensorAddress");
	}
	return slot;
}

void NNTensorRT::prepare_slots(const int slot_count)
{
	std::lock_guard<std::mutex> lock(slots_mutex);
	if (slot_count > profile_count)
	{
		throw std::out_of_range("NNTensorRT slot_count");
	}
	while ((int)slots.size() < slot_count)
	{
		slots.emplace_back(create_slot((int)slots.size()));
	}
}

NNTensorRT::InferenceSlot* NNTensorRT::get_slot(const int slot_id)
{
	if (slot_id < 0 || slot_id >= (int)slots.size())
	{
		throw std::out_of_range("NNTensorRT slot_id");
	}
	return slots[slot_id].get();
}

void NNTensorRT::forward(const int slot_id, const int batch_size, packed_features1_t* p1, packed_features2_t* p2, DType* y1, DType* y2)
{
	forward_impl(get_slot(slot_id), batch_size, p1, p2, y1, y2);
}

void NNTensorRT::forward_impl(InferenceSlot* slot, const int batch_size, packed_features1_t* p1, packed_features2_t* p2, DType* y1, DType* y2)
{
	checkCudaErrors(cudaMemcpyAsync(slot->p1_dev, p1, sizeof(packed_features1_t) * batch_size, cudaMemcpyHostToDevice, slot->stream));
	checkCudaErrors(cudaMemcpyAsync(slot->p2_dev, p2, sizeof(packed_features2_t) * batch_size, cudaMemcpyHostToDevice, slot->stream));
	unpack_features1(batch_size, slot->p1_dev, slot->x1_dev, slot->stream);
	unpack_features2(batch_size, slot->p2_dev, slot->x2_dev, slot->stream);
	checkCudaErrors(cudaGetLastError());

	auto dims1 = inputDims1;
	auto dims2 = inputDims2;
	dims1.d[0] = batch_size;
	dims2.d[0] = batch_size;
	if (!slot->context->setInputShape("input1", dims1) ||
		!slot->context->setInputShape("input2", dims2))
	{
		throw std::runtime_error("setInputShape");
	}

	if (!slot->context->enqueueV3(slot->stream))
	{
		throw std::runtime_error("enqueueV3");
	}

	checkCudaErrors(cudaMemcpyAsync(y1, slot->y1_dev, sizeof(DType) * MAX_MOVE_LABEL_NUM * (size_t)SquareNum * batch_size, cudaMemcpyDeviceToHost, slot->stream));
	checkCudaErrors(cudaMemcpyAsync(y2, slot->y2_dev, sizeof(DType) * batch_size, cudaMemcpyDeviceToHost, slot->stream));
	checkCudaErrors(cudaStreamSynchronize(slot->stream));
}

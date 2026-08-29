from libcpp.string cimport string
from libcpp cimport bool
from libcpp.utility cimport pair
from libcpp.vector cimport vector

import numpy as np
cimport numpy as np

import locale

cdef extern from "python_module.h" nogil:
    void init()
    void __hcpe_decode_with_value(const size_t len, char* ndhcpe, char* ndfeatures1, char* ndfeatures2, char* ndmove, char* ndresult, char* ndvalue)
    void __hcpe2_decode_with_value(const size_t len, char* ndhcpe2, char* ndfeatures1, char* ndfeatures2, char* ndmove, char* ndresult, char* ndvalue, char* ndaux)
    void __hcpe3_create_cache(const string& filepath)
    size_t __hcpe3_load_cache(const string& filepath) except +
    size_t __hcpe3_get_cache_num()
    size_t __load_hcpe3(const string& filepath, bool use_average, double a, double temperature, size_t& len)
    size_t __hcpe3_patch_with_hcpe(const string& filepath, size_t& add_len)
    void __hcpe3_decode_with_value(const size_t len, char* ndindex, char* ndfeatures1, char* ndfeatures2, char* ndprobability, char* ndresult, char* ndvalue, int decode_threads) except +
    size_t __load_evalfix(const string& filepath)
    void __hcpe3_get_hcpe(const size_t index, char* ndhcpe) except +
    void __hcpe3_prepare_evalfix(char* ndeval, char* ndresult)
    void __hcpe3_merge_cache(const string& file1, const string& file2, const string& out)
    void __hcpe3_concat_cache(const vector[string]& files, const string& out) except +
    void __hcpe3_split_cache(const string& file, const string& out, const size_t num_split) except +
    void __hcpe3_cache_write_start(const string& filepath, const size_t num) except +
    void __hcpe3_cache_write() except +
    void __hcpe3_cache_write_end() except +
    void __hcpe3_cache_re_eval(const size_t len, char* ndindex, char* ndlogits, char* ndvalue, const float alpha_p, const float alpha_v, const float alpha_r, const float dropoff, const int limit_candidates, const float temperature) except +
    void __hcpe3_reserve_train_data(unsigned int size)
    void __hcpe3_stat_cache() except +
    pair[int, int] __hcpe3_to_hcpe(const string& file1, const string& file2) except +
    pair[int, int] __hcpe3_clean(const string& file1, const string& file2) except +
    void __hcpe3_merge(const vector[string]& files, const string& out, const bool outMaxmove, const bool outMate, const bool outBrinkmate) except +
    unsigned int __get_max_features2_nyugyoku_num()

    void __fuseki_reset()
    int __fuseki_legal_drops(int* outPieceTypes, int* outSquares, int maxCount)
    void __fuseki_do_drop(int pieceType, int square)
    bool __fuseki_is_placement_done()
    int __fuseki_turn()
    int __fuseki_ply()
    int __fuseki_remaining(int color, int pieceType)
    string __fuseki_to_sfen()
    bool __fuseki_verify_final_sfen(const string& sfen)
    void __fuseki_make_input_features(char* ndfeatures1, char* ndfeatures2)
    int __fuseki_move_label(int pieceType, int square, int color)
    bool __fuseki_parse_usi_move(const string& moveStr, int* outPieceType, int* outSquare)
    void __make_input_features_from_sfen(const string& sfen, char* ndfeatures1, char* ndfeatures2)

init()

def hcpe_decode_with_value(np.ndarray ndhcpe, np.ndarray ndfeatures1, np.ndarray ndfeatures2, np.ndarray ndmove, np.ndarray ndresult, np.ndarray ndvalue):
    __hcpe_decode_with_value(len(ndhcpe), ndhcpe.data, ndfeatures1.data, ndfeatures2.data, ndmove.data, ndresult.data, ndvalue.data)

def hcpe2_decode_with_value(np.ndarray ndhcpe2, np.ndarray ndfeatures1, np.ndarray ndfeatures2, np.ndarray ndmove, np.ndarray ndresult, np.ndarray ndvalue, np.ndarray ndaux):
    __hcpe2_decode_with_value(len(ndhcpe2), ndhcpe2.data, ndfeatures1.data, ndfeatures2.data, ndmove.data, ndresult.data, ndvalue.data, ndaux.data)

def hcpe3_create_cache(str filepath):
    __hcpe3_create_cache(filepath.encode(locale.getpreferredencoding()))

def hcpe3_load_cache(str filepath):
    return __hcpe3_load_cache(filepath.encode(locale.getpreferredencoding()))

def hcpe3_get_cache_num():
    return __hcpe3_get_cache_num()

def load_hcpe3(str filepath, bool use_average, double a, double temperature):
    cdef size_t len = 0
    cdef size_t size = __load_hcpe3(filepath.encode(locale.getpreferredencoding()), use_average, a, temperature, len)
    return size, len

def hcpe3_patch_with_hcpe(str filepath):
    cdef size_t add_len = 0
    cdef size_t sum_len = __hcpe3_patch_with_hcpe(filepath.encode(locale.getpreferredencoding()), add_len)
    return sum_len, add_len

def hcpe3_decode_with_value(np.ndarray ndindex, np.ndarray ndfeatures1, np.ndarray ndfeatures2, np.ndarray ndprobability, np.ndarray ndresult, np.ndarray ndvalue, int decode_threads=2):
    __hcpe3_decode_with_value(len(ndindex), ndindex.data, ndfeatures1.data, ndfeatures2.data, ndprobability.data, ndresult.data, ndvalue.data, decode_threads)

def hcpe3_get_hcpe(size_t index, np.ndarray ndhcpe):
    __hcpe3_get_hcpe(index, ndhcpe.data)

def hcpe3_prepare_evalfix(str filepath):
    cdef size_t size = __load_evalfix(filepath.encode(locale.getpreferredencoding()))
    cdef np.ndarray ndeval = np.empty(size, np.int32)
    cdef np.ndarray ndresult = np.empty(size, np.float32)
    __hcpe3_prepare_evalfix(ndeval.data, ndresult.data)
    return ndeval, ndresult

def hcpe3_merge_cache(str file1, str file2, str out):
    __hcpe3_merge_cache(file1.encode(locale.getpreferredencoding()), file2.encode(locale.getpreferredencoding()), out.encode(locale.getpreferredencoding()))

def hcpe3_concat_cache(files, str out):
    cdef vector[string] cpp_files
    for filepath in files:
        cpp_files.push_back(str(filepath).encode(locale.getpreferredencoding()))
    __hcpe3_concat_cache(cpp_files, out.encode(locale.getpreferredencoding()))

def hcpe3_split_cache(str file, str out, size_t num_split):
    __hcpe3_split_cache(file.encode(locale.getpreferredencoding()), out.encode(locale.getpreferredencoding()), num_split)

def hcpe3_cache_write_start(str filepath, size_t num):
    __hcpe3_cache_write_start(filepath.encode(locale.getpreferredencoding()), num)

def hcpe3_cache_write():
    __hcpe3_cache_write()

def hcpe3_cache_write_end():
    __hcpe3_cache_write_end()

def hcpe3_cache_re_eval(np.ndarray ndindex, np.ndarray ndlogits, np.ndarray ndvalue, float alpha_p, float alpha_v, float alpha_r, float dropoff, int limit_candidates, float temperature=1.0):
    __hcpe3_cache_re_eval(len(ndindex), ndindex.data, ndlogits.data, ndvalue.data, alpha_p, alpha_v, alpha_r, dropoff, limit_candidates, temperature)

def hcpe3_reserve_train_data(unsigned int size):
    __hcpe3_reserve_train_data(size)

def hcpe3_stat_cache():
    __hcpe3_stat_cache()

def hcpe3_to_hcpe(str file1, str file2):
    return __hcpe3_to_hcpe(file1.encode(locale.getpreferredencoding()), file2.encode(locale.getpreferredencoding()))

def hcpe3_clean(str file1, str file2):
    return __hcpe3_clean(file1.encode(locale.getpreferredencoding()), file2.encode(locale.getpreferredencoding()))

def hcpe3_merge(files, str out, bool out_maxmove=False, bool out_mate=False, bool out_brinkmate=False):
    cdef vector[string] cpp_files
    for filepath in files:
        cpp_files.push_back(str(filepath).encode(locale.getpreferredencoding()))
    __hcpe3_merge(cpp_files, out.encode(locale.getpreferredencoding()), out_maxmove, out_mate, out_brinkmate)

def get_max_features2_nyugyoku_num():
    return __get_max_features2_nyugyoku_num()

# 布石将棋の布石フェーズ（グローバルに1局面だけ保持する軽量な状態）
FUSEKI_MAX_MOVES = 400

def fuseki_reset():
    __fuseki_reset()

def fuseki_legal_drops():
    cdef np.ndarray ndPieceTypes = np.empty(FUSEKI_MAX_MOVES, dtype=np.int32)
    cdef np.ndarray ndSquares = np.empty(FUSEKI_MAX_MOVES, dtype=np.int32)
    cdef int n = __fuseki_legal_drops(<int*>ndPieceTypes.data, <int*>ndSquares.data, FUSEKI_MAX_MOVES)
    return [(int(ndPieceTypes[i]), int(ndSquares[i])) for i in range(n)]

def fuseki_do_drop(int piece_type, int square):
    __fuseki_do_drop(piece_type, square)

def fuseki_is_placement_done():
    return __fuseki_is_placement_done()

def fuseki_turn():
    return __fuseki_turn()

def fuseki_ply():
    return __fuseki_ply()

def fuseki_remaining(int color, int piece_type):
    return __fuseki_remaining(color, piece_type)

def fuseki_to_sfen():
    return __fuseki_to_sfen().decode('ascii')

def fuseki_verify_final_sfen(str sfen):
    return __fuseki_verify_final_sfen(sfen.encode('ascii'))

# ndfeatures1/ndfeatures2は呼び出し側があらかじめ確保する
# （dlshogi.common.FEATURES1_NUM/FEATURES2_NUMに基づく(*, 9, 9)形状。他のdecode系関数と同じ規約）。
def fuseki_make_input_features(np.ndarray ndfeatures1, np.ndarray ndfeatures2):
    __fuseki_make_input_features(ndfeatures1.data, ndfeatures2.data)

def fuseki_move_label(int piece_type, int square, int color):
    return __fuseki_move_label(piece_type, square, color)

# usi/main.cppがbestmoveとして返す"K*5i"形式の指し手文字列を(piece_type, square)に変換する。
# 不正な形式（またはUSIの投了/入玉宣言など駒打ちでないトークン）の場合はNoneを返す。
def fuseki_parse_usi_move(str move_str):
    cdef int piece_type = 0
    cdef int square = 0
    if not __fuseki_parse_usi_move(move_str.encode('ascii'), &piece_type, &square):
        return None
    return piece_type, square

def make_input_features_from_sfen(str sfen, np.ndarray ndfeatures1, np.ndarray ndfeatures2):
    __make_input_features_from_sfen(sfen.encode('ascii'), ndfeatures1.data, ndfeatures2.data)

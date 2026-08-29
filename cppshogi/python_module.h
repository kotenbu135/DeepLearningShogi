#pragma once

#include <string>
#include <vector>

void init();
void __hcpe_decode_with_value(const size_t len, char* ndhcpe, char* ndfeatures1, char* ndfeatures2, char* ndmove, char* ndresult, char* ndvalue);
void __hcpe2_decode_with_value(const size_t len, char* ndhcpe2, char* ndfeatures1, char* ndfeatures2, char* ndmove, char* ndresult, char* ndvalue, char* ndaux);
void __hcpe3_create_cache(const std::string& filepath);
size_t __hcpe3_load_cache(const std::string& filepath);
size_t __hcpe3_get_cache_num();
size_t __load_hcpe3(const std::string& filepath, bool use_average, double a, double temperature, size_t& len);
size_t __hcpe3_patch_with_hcpe(const std::string& filepath, size_t& add_len);
void __hcpe3_decode_with_value(const size_t len, char* ndindex, char* ndfeatures1, char* ndfeatures2, char* ndprobability, char* ndresult, char* ndvalue, int decode_threads=2);
size_t __load_evalfix(const std::string& filepath);
void __hcpe3_get_hcpe(const size_t index, char* ndhcpe);
void __hcpe3_prepare_evalfix(char* ndeval, char* ndresult);
void __hcpe3_merge_cache(const std::string& file1, const std::string& file2, const std::string& out);
void __hcpe3_concat_cache(const std::vector<std::string>& files, const std::string& out);
void __hcpe3_split_cache(const std::string& file, const std::string& out, const size_t num_split);
void __hcpe3_cache_write_start(const std::string& filepath, const size_t num);
void __hcpe3_cache_write();
void __hcpe3_cache_write_end();
void __hcpe3_cache_re_eval(const size_t len, char* ndindex, char* ndlogits, char* ndvalue, const float alpha_p, const float alpha_v, const float alpha_r, const float dropoff, const int limit_candidates, const float temperature=1.0f);
void __hcpe3_reserve_train_data(unsigned int size);
void __hcpe3_stat_cache();
std::pair<int, int> __hcpe3_to_hcpe(const std::string& file1, const std::string& file2);
std::pair<int, int> __hcpe3_clean(const std::string& file1, const std::string& file2);
void __hcpe3_merge(const std::vector<std::string>& files, const std::string& out, const bool outMaxmove=false, const bool outMate=false, const bool outBrinkmate=false);
unsigned int __get_max_features2_nyugyoku_num();

// 布石将棋の布石フェーズ（グローバルに1局面だけ保持する）
void __fuseki_reset();
// legalDropsをoutPieceTypes/outSquaresに書き出し、手数を返す。書き出し先の容量はmaxCount。
int __fuseki_legal_drops(int* outPieceTypes, int* outSquares, int maxCount);
void __fuseki_do_drop(int pieceType, int square);
bool __fuseki_is_placement_done();
int __fuseki_turn();
int __fuseki_ply();
int __fuseki_remaining(int color, int pieceType);
std::string __fuseki_to_sfen();
// sfenを実際にPosition::set()で読み込み、王1枚ずつ・盤上40枚・持ち駒0・41手目、を満たすか検証する。
bool __fuseki_verify_final_sfen(const std::string& sfen);

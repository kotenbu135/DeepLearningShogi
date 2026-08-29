// 布石将棋（https://shogitter.com/rule/布石将棋）の布石フェーズ専用の軽量な局面表現。
//
// 布石フェーズ（開始から双方20手・計40手）は shogitter のルール上
// 「双方20手指すまでは（王手・詰みの）判定をしない」ため、Apery/dlshogi 本来の
// Position クラスが前提とする「玉は必ず盤上に1枚ある」という不変条件に一切触れずに
// 実装できる。そのため Hand / Move / Position には依存しない完全に独立した状態として
// 実装し、40手完了時点で SFEN 文字列を組み立てて Position::set() に渡す。
#ifndef APERY_FUSEKI_HPP
#define APERY_FUSEKI_HPP

#include "color.hpp"
#include "piece.hpp"
#include "square.hpp"
#include <string>
#include <utility>
#include <vector>

class FusekiPosition {
public:
    static const int TotalPlies = 40; // 双方20手ずつ

    FusekiPosition() { reset(); }

    void reset();

    // 手番側が指せる合法な「駒種を打つ」手を列挙する。
    std::vector<std::pair<PieceType, Square>> legalDrops() const;

    // 駒を打つ。pt, sq は legalDrops() が返したものであること（呼び出し側で検証済みの前提）。
    void doDrop(PieceType pt, Square sq);

    bool isPlacementDone() const { return ply_ >= TotalPlies; }
    Color turn() const { return turn_; }
    int ply() const { return ply_; }
    int remaining(Color c, PieceType pt) const { return remaining_[c][pt]; }
    Piece pieceOn(Square sq) const { return board_[sq]; }

    // 40手完了後に呼ぶ。通常の Position::set(sfen) にそのまま渡せる文字列を返す。
    std::string toSFEN() const;

private:
    // 手番側 us にとって、指定の筋 file に非歩の駒を打った結果、
    // 自陣4段(own camp)のその筋が非歩の駒で埋まってしまうか（禁じ手）。
    bool wouldFillFileWithNonPawn(Color us, File file, Rank dropRank) const;

    Piece board_[SquareNum];
    int remaining_[ColorNum][King + 1]; // Pawn, Lance, Knight, Silver, Gold, Bishop, Rook, King の残数
    Color turn_;
    int ply_;
};

// 布石フェーズの手を USI 風の駒打ち表記（例: "K*5i"）に変換する。
// 通常の Move::toUSI() 同様、打つ駒の文字は手番に関わらず大文字。
std::string fusekiMoveToUSI(PieceType pt, Square sq);

// 上記の逆変換。フォーマット不正なら false を返す。
bool parseFusekiMoveUSI(const std::string& moveStr, PieceType& pt, Square& sq);

#endif // #ifndef APERY_FUSEKI_HPP

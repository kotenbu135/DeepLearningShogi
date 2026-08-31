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
    //
    // 最終手（40手目 = 後手の20手目）では、打った結果**自玉が相手の利きに当たる**手を除外する
    // （通常将棋の「自玉を王手に晒せない」に揃えたルール。docs/rules.md参照）。41手目は先手番
    // なので、後手が自玉を取られる形で置くと先手が玉を取って即終局してしまうため。
    // 先手の最終手（39手目）にはこの制限は掛からない（41手目に先手が王手されているのは
    // 通常の王手であって非合法ではない）。
    //
    // 例外: 39手目の時点で既に後手玉が王手されており、残り1枚では合流できない場合
    // （桂の王手など）、除外すると合法手が0になる。そのときは制限を外して従来通り列挙する
    // （movegenが手詰まりにならないという不変条件を保つため。docs/rules.md参照）。
    // その局面は41手目に先手が玉を取れる状態なので、対局ドライバが「手番側が玉を取れるなら
    // 手番側の勝ち」と裁定する。
    std::vector<std::pair<PieceType, Square>> legalDrops() const;

    // 色cの玉が相手の利きに当たっているか。玉がまだ盤上に無ければfalse。
    bool isKingAttacked(Color c) const;

    // 色cの玉のマス。まだ打っていなければ SquareNum。
    Square kingSquare(Color c) const;

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

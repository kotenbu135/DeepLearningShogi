#include "fuseki.hpp"
#include <cctype>

namespace {
    // 手番 us の自陣（四段目以内）の範囲を返す。
    // DefaultStartPositionSFEN の並び（後手が1段目〜9段目のうち1段目側、先手が9段目側）より、
    // 後手(White)の自陣は1段目〜4段目、先手(Black)の自陣は6段目〜9段目。
    Rank campBegin(const Color us) { return (us == Black ? Rank6 : Rank1); }
    Rank campEnd(const Color us)   { return (us == Black ? Rank9 : Rank4); }

    const PieceType DroppablePieceTypes[] = {
        Pawn, Lance, Knight, Silver, Gold, Bishop, Rook, King
    };
}

void FusekiPosition::reset() {
    for (Square sq = SQ11; sq < SquareNum; ++sq)
        board_[sq] = Empty;

    for (Color c = Black; c < ColorNum; ++c) {
        remaining_[c][Pawn]   = 9;
        remaining_[c][Lance]  = 2;
        remaining_[c][Knight] = 2;
        remaining_[c][Silver] = 2;
        remaining_[c][Gold]   = 2;
        remaining_[c][Bishop] = 1;
        remaining_[c][Rook]   = 1;
        remaining_[c][King]   = 1;
    }

    turn_ = Black;
    ply_ = 0;
}

bool FusekiPosition::wouldFillFileWithNonPawn(const Color us, const File file, const Rank dropRank) const {
    const Rank begin = campBegin(us);
    const Rank end = campEnd(us);
    int occupiedNonPawnOthers = 0;

    for (Rank r = begin; r <= end; ++r) {
        if (r == dropRank)
            continue; // これから打つマス（現在は空のはず）
        const Piece pc = board_[makeSquare(file, r)];
        if (pc != Empty && pieceToPieceType(pc) != Pawn)
            ++occupiedNonPawnOthers;
    }

    // 自陣4マスのうち、これから打つマス以外の3マスすべてが既に非歩で埋まっていれば、
    // この非歩の駒を打つことで4マス全てが非歩になってしまう＝禁じ手。
    const int otherSquares = static_cast<int>(end) - static_cast<int>(begin);
    return occupiedNonPawnOthers == otherSquares;
}

std::vector<std::pair<PieceType, Square>> FusekiPosition::legalDrops() const {
    std::vector<std::pair<PieceType, Square>> moves;
    const Color us = turn_;
    const Rank begin = campBegin(us);
    const Rank end = campEnd(us);

    for (const PieceType pt : DroppablePieceTypes) {
        if (remaining_[us][pt] <= 0)
            continue;

        for (File file = File1; file < FileNum; ++file) {
            if (pt == Pawn) {
                // 二歩回避: 自陣内の同じ筋に既に自分の歩があれば、この筋には打てない。
                bool hasOwnPawnInFile = false;
                for (Rank r = begin; r <= end; ++r) {
                    const Piece pc = board_[makeSquare(file, r)];
                    if (pc != Empty && pieceToColor(pc) == us && pieceToPieceType(pc) == Pawn) {
                        hasOwnPawnInFile = true;
                        break;
                    }
                }
                if (hasOwnPawnInFile)
                    continue;
            }

            for (Rank r = begin; r <= end; ++r) {
                const Square sq = makeSquare(file, r);
                if (board_[sq] != Empty)
                    continue;
                if (pt != Pawn && wouldFillFileWithNonPawn(us, file, r))
                    continue;
                moves.emplace_back(pt, sq);
            }
        }
    }

    return moves;
}

void FusekiPosition::doDrop(const PieceType pt, const Square sq) {
    board_[sq] = colorAndPieceTypeToPiece(turn_, pt);
    --remaining_[turn_][pt];
    ++ply_;
    turn_ = oppositeColor(turn_);
}

std::string FusekiPosition::toSFEN() const {
    static const char* PieceTypeToCharTable[King + 1] = {
        "", "P", "L", "N", "S", "B", "R", "G", "K"
    };

    std::string sfen;
    for (Rank r = Rank1; r < RankNum; ++r) {
        if (r != Rank1)
            sfen += '/';

        int emptyCount = 0;
        for (File f = File9; f >= File1; --f) {
            const Piece pc = board_[makeSquare(f, r)];
            if (pc == Empty) {
                ++emptyCount;
                continue;
            }
            if (emptyCount > 0) {
                sfen += std::to_string(emptyCount);
                emptyCount = 0;
            }
            std::string ch = PieceTypeToCharTable[pieceToPieceType(pc)];
            if (pieceToColor(pc) == White) {
                for (char& c : ch)
                    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            }
            sfen += ch;
        }
        if (emptyCount > 0)
            sfen += std::to_string(emptyCount);
    }

    sfen += ' ';
    sfen += (turn_ == Black ? 'b' : 'w');
    sfen += " - "; // 布石完了時点で両陣とも持ち駒は全て打ち終えている
    sfen += std::to_string(ply_ + 1);
    return sfen;
}

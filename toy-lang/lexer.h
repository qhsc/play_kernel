#pragma once
#include <string>

// Lexer is reponsible for tokenizing the input code.
// Break input text into a sequence of tokens.

namespace lexer {

enum class Token {
    EOF_ = -1,

    // commands
    DEF = -2,
    EXTERN = -3,

    // primary
    IDENTIFIER = -4,
    NUMBER = -5,
};

// global states to hold token value
static std::string IdentifierStr;
static double NumVal;

static int gettok() {
    static int lastChar = ' ';
    // skip whitespace
    while (isspace(lastChar)) {
        lastChar = getchar();
    }

    // identifier
    if (isalpha(lastChar)) {
        IdentifierStr = lastChar;
        while (isalnum((lastChar = getchar()))) {
            IdentifierStr += lastChar;
        }
        if (IdentifierStr == "def") {
            return int(Token::DEF);
        }
        if (IdentifierStr == "extern") {
            return int(Token::EXTERN);
        }
        return int(Token::IDENTIFIER);
    }

    // number
    if (isdigit(lastChar) or lastChar == '.') {
        std::string numStr;
        do {
            numStr += lastChar;
            lastChar = getchar();
        } while (isdigit(lastChar) or lastChar == '.');
        NumVal = strtod(numStr.c_str(), nullptr);
        return int(Token::NUMBER);
    }

    // eof
    if (lastChar == EOF) {
        return int(Token::EOF_);
    }

    // unknown, just return it and move on.
    // note: move on is necessary, otherwise we will stuck in a loop.
    auto currentChar = lastChar;
    lastChar = getchar();
    return currentChar;
}

}  // namespace lexer
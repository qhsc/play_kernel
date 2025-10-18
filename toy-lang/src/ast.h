#pragma once

#include <cassert>
#include <cctype>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "lexer.h"

#include <llvm/IR/IRBuilder.h>
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"

namespace ast {

static int CurTok;
static int getNextToken() {
    return CurTok = lexer::gettok();
}

class ExprAST {
   public:
    virtual ~ExprAST() = default;

    virtual std::string dump() const = 0;

    virtual llvm::Value *codegen() = 0;

    static std::unique_ptr<ExprAST> LogError(const char *str) {
        fprintf(stderr, "Error: %s\n", str);
        return nullptr;
    }

    static std::unique_ptr<ExprAST> parse();
};

class NumberExprAST : public ExprAST {
    double val_;

   public:
    NumberExprAST(double val) : val_(val) {
    }

    std::string dump() const override {
        return "num:" + std::to_string(val_);
    }

    static std::unique_ptr<NumberExprAST> parse() {
        assert(CurTok == int(lexer::Token::NUMBER));
        auto ret = std::make_unique<NumberExprAST>(lexer::NumVal);
        getNextToken();
        return ret;
    }

    llvm::Value *codegen() override;
};

class VariableExprAST : public ExprAST {
    std::string name_;

   public:
    VariableExprAST(std::string name) : name_(std::move(name)) {
    }

    std::string dump() const override {
        return "var:" + name_;
    }

    llvm::Value *codegen() override;
};

class BinaryExprAST : public ExprAST {
    char op_;
    std::unique_ptr<ExprAST> lhs_;
    std::unique_ptr<ExprAST> rhs_;

   public:
    BinaryExprAST(char op, std::unique_ptr<ExprAST> lhs, std::unique_ptr<ExprAST> rhs)
        : op_(op), lhs_(std::move(lhs)), rhs_(std::move(rhs)) {
    }

    static int getPrecedence(int token) {
        static const std::map<char, int> precedence_{
            {'<', 10}, {'+', 20}, {'-', 20}, {'*', 40}, {'/', 40},
        };
        if (!isascii(token) || precedence_.find(token) == precedence_.end()) {
            return -1;
        }
        return precedence_.at(token);
    }

    std::string dump() const override {
        return "(binop:" + std::string(1, op_) + " " + lhs_->dump() + ", " + rhs_->dump() + ")";
    }

    llvm::Value *codegen() override;
};

class CallExprAST : public ExprAST {
    std::string callee_;
    std::vector<std::unique_ptr<ExprAST>> args_;

   public:
    CallExprAST(const std::string &callee, std::vector<std::unique_ptr<ExprAST>> args)
        : callee_(callee), args_(std::move(args)) {
    }

    std::string dump() const override {
        std::string args_str;
        for (const auto &arg : args_) {
            args_str += arg->dump() + ", ";
        }
        args_str = args_str.substr(0, args_str.size() - 2);
        return "(call:" + callee_ + " " + args_str + ")";
    }

    llvm::Value *codegen() override;
};

/// PrototypeAST - This class represents the "prototype" for a function,
/// which captures its name, and its argument names (thus implicitly the number
/// of arguments the function takes).
class PrototypeAST {
    std::string name_;
    std::vector<std::string> args_;

   public:
    PrototypeAST(std::string name, std::vector<std::string> args) : name_(std::move(name)), args_(std::move(args)) {
    }

    std::string dump() const {
        std::string args_str;
        for (const auto &arg : args_) {
            args_str += arg + ", ";
        }
        args_str = args_str.substr(0, args_str.size() - 2);
        return "(proto:" + name_ + " " + args_str + ")";
    }

    const std::string &getName() const {
        return name_;
    }

    static std::unique_ptr<PrototypeAST> LogErrorP(const char *Str) {
        ExprAST::LogError(Str);
        return nullptr;
    }

    static std::unique_ptr<PrototypeAST> parse() {
        if (CurTok != int(lexer::Token::IDENTIFIER)) {
            return LogErrorP("Expected function name in function definition");
        }

        auto name = lexer::IdentifierStr;
        getNextToken();

        if (CurTok != '(') {
            return LogErrorP("Expected '(' in function definition");
        }

        std::vector<std::string> args;
        while (getNextToken() == int(lexer::Token::IDENTIFIER)) {
            args.push_back(lexer::IdentifierStr);  // the args shuold splited by space
        }
        if (CurTok != ')') {
            return LogErrorP("Expected ')' in function definition");
        }
        getNextToken();
        return std::make_unique<PrototypeAST>(name, std::move(args));
    }

    llvm::Function *codegen();
};

/// FunctionAST - This class represents a function definition itself.
class FunctionAST {
    std::unique_ptr<PrototypeAST> proto_;
    std::unique_ptr<ExprAST> body_;

   public:
    FunctionAST(std::unique_ptr<PrototypeAST> proto, std::unique_ptr<ExprAST> body)
        : proto_(std::move(proto)), body_(std::move(body)) {
    }
    std::string dump() const {
        return "(def:" + proto_->dump() + " " + body_->dump() + ")";
    }

    /// definition ::= 'def' prototype expression
    static std::unique_ptr<FunctionAST> parseDefinition() {
        getNextToken();  // eat def.
        auto proto = PrototypeAST::parse();
        if (proto == nullptr) {
            return nullptr;
        }
        if (auto body = ExprAST::parse()) {
            return std::make_unique<FunctionAST>(std::move(proto), std::move(body));
        }
        return nullptr;
    }

    /// external ::= 'extern' prototype
    static std::unique_ptr<PrototypeAST> parseExtern() {
        getNextToken();  // eat extern.
        return PrototypeAST::parse();
    }

    llvm::Function *codegen();
};

static std::unique_ptr<ExprAST> parseParenExpr() {
    assert(CurTok == '(');
    getNextToken();

    auto expr = ExprAST::parse();
    if (expr == nullptr) {
        return nullptr;
    }

    if (CurTok != ')') {
        return ExprAST::LogError("parseParenExpr: expected ')'");
    }
    getNextToken();
    return expr;
}

static std::unique_ptr<ExprAST> parseIdentifierExpr() {
    assert(CurTok == int(lexer::Token::IDENTIFIER));
    auto name = lexer::IdentifierStr;
    getNextToken();

    if (CurTok != '(') {
        return std::make_unique<VariableExprAST>(name);
    }
    getNextToken();  // consume '('

    std::vector<std::unique_ptr<ExprAST>> args;
    if (CurTok != ')') {
        while (true) {
            auto arg = ExprAST::parse();
            if (arg == nullptr) {
                return nullptr;
            }
            args.push_back(std::move(arg));
            if (CurTok == ')') {
                break;
            }
            if (CurTok != ',') {
                return ExprAST::LogError("parseIdentifierExpr Args: expected ',' or ')'");
            }
            getNextToken();  // consume ','
        }
    }

    getNextToken();  // consume ')'
    return std::make_unique<CallExprAST>(name, std::move(args));
}

/// primary
///   ::= identifierexpr
///   ::= numberexpr
///   ::= parenexpr
static std::unique_ptr<ExprAST> parsePrimary() {
    switch (CurTok) {
    default:
        return ExprAST::LogError("unknown token when expecting an expression");
    case int(lexer::Token::IDENTIFIER):
        // variable or function call
        return parseIdentifierExpr();
    case int(lexer::Token::NUMBER):
        // number
        return NumberExprAST::parse();
    case '(':
        // parse what inside the parentheses
        return parseParenExpr();
    }
}

/// binoprhs
///   ::= ('+' primary)*
static std::unique_ptr<ExprAST> parseBinOpRHS(int exprPrec, std::unique_ptr<ExprAST> lhs) {
    while (true) {
        int tokenPrec = BinaryExprAST::getPrecedence(CurTok);

        // If this is a binop that binds at least as tightly as the current binop,
        // consume it, otherwise we are done.
        if (tokenPrec < exprPrec) {
            return lhs;
        }

        int binOp = CurTok;
        getNextToken();

        // Parse the primary expression after the binary operator.
        auto rhs = parsePrimary();
        if (rhs == nullptr) {
            return nullptr;
        }

        // If BinOp binds less tightly with RHS than the operator after RHS, let
        // the pending operator take RHS as its LHS.
        int nextPrec = BinaryExprAST::getPrecedence(CurTok);
        if (nextPrec > tokenPrec) {
            rhs = parseBinOpRHS(tokenPrec + 1, std::move(rhs));
            if (rhs == nullptr) {
                return nullptr;
            }
        }

        // merge lhs and rhs as BinaryExprAST and as lhs for next iteration
        lhs = std::make_unique<BinaryExprAST>(binOp, std::move(lhs), std::move(rhs));
    }
}

inline std::unique_ptr<ExprAST> ExprAST::parse() {
    auto lhs = parsePrimary();
    if (lhs == nullptr) {
        return nullptr;
    }
    return parseBinOpRHS(0, std::move(lhs));
}

/// toplevelexpr ::= expression
static std::unique_ptr<FunctionAST> ParseTopLevelExpr() {
    if (auto E = ExprAST::parse()) {
        // Make an anonymous proto.
        auto Proto = std::make_unique<PrototypeAST>("__anon_expr", std::vector<std::string>());
        return std::make_unique<FunctionAST>(std::move(Proto), std::move(E));
    }
    return nullptr;
}

//===----------------------------------------------------------------------===//
// Top-Level parsing
//===----------------------------------------------------------------------===//

static void HandleDefinition() {
    if (auto def = FunctionAST::parseDefinition()) {
        fprintf(stderr, "Parsed a function definition.\n");
        fprintf(stderr, "%s\n", def->dump().c_str());
    } else {
        // Skip token for error recovery.
        getNextToken();
    }
}

static void HandleExtern() {
    if (auto extern_ = FunctionAST::parseExtern()) {
        fprintf(stderr, "Parsed an extern\n");
        fprintf(stderr, "%s\n", extern_->dump().c_str());
    } else {
        // Skip token for error recovery.
        getNextToken();
    }
}

static void HandleTopLevelExpression() {
    // Evaluate a top-level expression into an anonymous function.
    if (auto top_level_expr = ParseTopLevelExpr()) {
        fprintf(stderr, "Parsed a top-level expr\n");
        fprintf(stderr, "%s\n", top_level_expr->dump().c_str());
    } else {
        // Skip token for error recovery.
        getNextToken();
    }
}

/// top ::= definition | external | expression | ';'
static void MainLoop() {
    while (true) {
        fprintf(stderr, "ready> ");
        ast::getNextToken();
        switch (CurTok) {
        case int(lexer::Token::EOF_):
            return;
        case ';':  // ignore top-level semicolons.
            getNextToken();
            break;
        case int(lexer::Token::DEF):
            HandleDefinition();
            break;
        case int(lexer::Token::EXTERN):
            HandleExtern();
            break;
        default:
            HandleTopLevelExpression();
            break;
        }
    }
}

}  // namespace ast
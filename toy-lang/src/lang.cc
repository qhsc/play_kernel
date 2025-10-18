#include <cctype>
#include <cstdio>

#include "ast.h"

/*

# Compute the x'th fibonacci number.
def fib(x)
  if x < 3 then
    1
  else
    fib(x-1)+fib(x-2)

# This expression will compute the 40th number.
fib(40)

FunctionAST
    * PrototypeAST (prototype)
        * name
        * args
    * ExprAST (body)
        * NumberExprAST
        * VariableExprAST
        * BinaryExprAST
        * CallExprAST

Parser:
ExparAST::parse()
    * parsePrimary()
        * parseNum
        * parseIdentifier
            * parseVariable
            * parseCall
                * name
                * args
                    * ExprAST::parse()
        * parseParen
            * ExprAST::parse()
    * parseBinOpRHS
        * parsePrimary
        * parseBinOpRHS
        * merge lhs and rhs
*/

int main() {
    // // Prime the first token.
    // fprintf(stderr, "ready> ");
    // ast::getNextToken();

    // Run the main "interpreter loop" now.
    ast::MainLoop();
    return 0;
}
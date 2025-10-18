#include "ast.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Value.h"

static std::unique_ptr<llvm::LLVMContext> TheContext;
static std::unique_ptr<llvm::IRBuilder<>> Builder;
static std::unique_ptr<llvm::Module> TheModule;
static std::map<std::string, llvm::Value *> NamedValues;

llvm::Value *LogErrorV(const char *str) {
    ast::ExprAST::LogError(str);
    return nullptr;
}

// Simple implementations to satisfy the virtual function requirements
llvm::Value *ast::NumberExprAST::codegen() {
    return llvm::ConstantFP::get(*TheContext, llvm::APFloat(val_));
}

llvm::Value *ast::VariableExprAST::codegen() {
    llvm::Value *v = NamedValues[name_];
    if (!v) {
        return LogErrorV("Unknown variable name");
    }
    return v;
}

llvm::Value *ast::BinaryExprAST::codegen() {
    llvm::Value *l = lhs_->codegen();
    if (!l) {
        return nullptr;
    }
    llvm::Value *r = rhs_->codegen();
    if (!r) {
        return nullptr;
    }
    switch (op_) {
    case '+':
        return Builder->CreateFAdd(l, r, "addtmp");
    case '-':
        return Builder->CreateFSub(l, r, "subtmp");
    case '*':
        return Builder->CreateFMul(l, r, "multmp");
    case '/':
        return Builder->CreateFDiv(l, r, "divtmp");
    case '<':
        l = Builder->CreateFCmpULT(l, r, "cmptmp");
        // Convert bool 0/1 to double 0.0 or 1.0
        return Builder->CreateUIToFP(l, llvm::Type::getDoubleTy(*TheContext), "booltmp");
    default:
        return LogErrorV("invalid binary operator");
    }
}

llvm::Value *ast::CallExprAST::codegen() {
    llvm::Function *calleeF = TheModule->getFunction(callee_);
    if (!calleeF) {
        return LogErrorV("Unknown function referenced");
    }
    if (calleeF->arg_size() != args_.size()) {
        return LogErrorV("Incorrect number of arguments passed");
    }
    std::vector<llvm::Value *> argsV(args_.size());
    for (unsigned i = 0, e = args_.size(); i != e; ++i) {
        argsV[i] = args_[i]->codegen();
        if (!argsV[i]) {
            return nullptr;
        }
    }
    return Builder->CreateCall(calleeF, argsV);
}

llvm::Function *ast::PrototypeAST::codegen() {
    std::vector<llvm::Type *> doubles(args_.size(), llvm::Type::getDoubleTy(*TheContext));
    llvm::FunctionType *FT = llvm::FunctionType::get(llvm::Type::getDoubleTy(*TheContext), doubles, false);
    llvm::Function *F = llvm::Function::Create(FT, llvm::Function::ExternalLinkage, name_, *TheModule);
    unsigned idx = 0;
    for (auto &arg : F->args()) {
        arg.setName(args_[idx++]);
    }
    return F;
}

llvm::Function *ast::FunctionAST::codegen() {
    llvm::Function *f = TheModule->getFunction(proto_->getName());
    if (!f) {
        f = proto_->codegen();
    }
    if (!f) {
        return nullptr;
    }
    if (!f->empty()) {
        return (llvm::Function *)LogErrorV("Function cannot be redefined");
    }

    llvm::BasicBlock *bb = llvm::BasicBlock::Create(*TheContext, "entry", f);
    Builder->SetInsertPoint(bb);
    NamedValues.clear();
    for (auto &arg : f->args()) {
        NamedValues[std::string(arg.getName())] = &arg;
    }
    
    if (llvm::Value *ret_val = body_->codegen()) {
        Builder->CreateRet(ret_val);
        verifyFunction(*f);
        return f;
    }
    f->eraseFromParent();
    return nullptr;
}
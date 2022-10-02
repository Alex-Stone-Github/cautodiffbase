#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

typedef struct { // # homebrew vtable
    void* evaluate_func_p;
    void* requires_grad_func_p;
    void* base;
} l_ComputeNode;
typedef struct { // this basic fields stored inside of a leaf
    double value;
    bool requires_grad;
} l_LeafBase;
typedef struct { // this stores the basic fields needed for an operation
    l_ComputeNode* pa;
    l_ComputeNode* pb;
} l_OperationBase;

double l_evaluate(l_ComputeNode* node); // evaluates a computation node
bool l_requires_grad(l_ComputeNode* node); // evaluates if a computation node needs a gradient
l_ComputeNode* l_new_leaf(double value, bool requires_grad);
void l_delete_leaf(l_ComputeNode* leaf);
void l_delete_operation(l_ComputeNode* operation);
l_ComputeNode* l_new_addition(l_ComputeNode* pa, l_ComputeNode* pb);
l_ComputeNode* l_new_subtraction(l_ComputeNode* pa, l_ComputeNode* pb);
l_ComputeNode* l_new_multiplication(l_ComputeNode* pa, l_ComputeNode* pb);
l_ComputeNode* l_new_division(l_ComputeNode* pa, l_ComputeNode* pb);


static double leaf_evalute(void* base);
static double leaf_requires_grad(void* base);
static double operation_requires_grad(void* base);
static double add_evaluate(void* base);
static double sub_evaluate(void* base);
static double mul_evaluate(void* base);
static double div_evaluate(void* base);


#include "lib.h"

double l_evaluate(l_ComputeNode* node) {
    double (*node_evaluate)(void*) = node->evaluate_func_p;
    return node_evaluate(node->base);
}
bool l_requires_grad(l_ComputeNode* node) {
    double (*node_requires_grad_func)(void*) = node->requires_grad_func_p;
    return node_requires_grad_func(node->base);
}
static double leaf_evaluate(void* base) {
    l_LeafBase* leaf_base = (l_LeafBase*)base;
    return leaf_base->value;
}
static double leaf_requires_grad(void* base) {
    l_LeafBase* leaf_base = (l_LeafBase*)base;
    return leaf_base->requires_grad;
}
static double operation_requires_grad(void* base) {
    l_OperationBase* operation_base = (l_OperationBase*)base;
    return l_requires_grad(operation_base->pa) || l_requires_grad(operation_base->pb);
}
static double add_evaluate(void* base) {
    l_OperationBase* operation_base = (l_OperationBase*)base;
    return l_evaluate(operation_base->pa) + l_evaluate(operation_base->pb);
}
static double sub_evaluate(void* base) {
    l_OperationBase* operation_base = (l_OperationBase*)base;
    return l_evaluate(operation_base->pa) - l_evaluate(operation_base->pb);
}
static double mul_evaluate(void* base) {
    l_OperationBase* operation_base = (l_OperationBase*)base;
    return l_evaluate(operation_base->pa) * l_evaluate(operation_base->pb);
}
static double div_evaluate(void* base) {
    l_OperationBase* operation_base = (l_OperationBase*)base;
    return l_evaluate(operation_base->pa) / l_evaluate(operation_base->pb);
}
l_ComputeNode* l_new_leaf(double value, bool requires_grad) {
    l_LeafBase* leaf_base = malloc(sizeof(l_LeafBase));
    leaf_base->value = value;
    leaf_base->requires_grad = requires_grad;
    l_ComputeNode* leaf = malloc(sizeof(l_ComputeNode));
    leaf->evaluate_func_p = leaf_evaluate;
    leaf->requires_grad_func_p = leaf_requires_grad;
    leaf->base = (void*)leaf_base;
    return leaf;
}
void l_delete_leaf(l_ComputeNode* leaf) {
    free(leaf->base);
    free(leaf);
}
l_ComputeNode* l_new_addition(l_ComputeNode* pa, l_ComputeNode* pb) {
    l_OperationBase* operation_base = malloc(sizeof(l_OperationBase));
    operation_base->pa = pa;
    operation_base->pb = pb;
    l_ComputeNode* addition = malloc(sizeof(l_ComputeNode));
    addition->evaluate_func_p = add_evaluate;
    addition->requires_grad_func_p = operation_requires_grad;
    addition->base = (void*)operation_base;
    return addition;
}
void l_delete_operation(l_ComputeNode* operation) {
    free(operation);
}
l_ComputeNode* l_new_subtraction(l_ComputeNode* pa, l_ComputeNode* pb) {
    l_OperationBase* operation_base = malloc(sizeof(l_OperationBase));
    operation_base->pa = pa;
    operation_base->pb = pb;
    l_ComputeNode* addition = malloc(sizeof(l_ComputeNode));
    addition->evaluate_func_p = sub_evaluate;
    addition->requires_grad_func_p = operation_requires_grad;
    addition->base = (void*)operation_base;
    return addition;
}
l_ComputeNode* l_new_multiplication(l_ComputeNode* pa, l_ComputeNode* pb) {
    l_OperationBase* operation_base = malloc(sizeof(l_OperationBase));
    operation_base->pa = pa;
    operation_base->pb = pb;
    l_ComputeNode* addition = malloc(sizeof(l_ComputeNode));
    addition->evaluate_func_p = mul_evaluate;
    addition->requires_grad_func_p = operation_requires_grad;
    addition->base = (void*)operation_base;
    return addition;
}
l_ComputeNode* l_new_division(l_ComputeNode* pa, l_ComputeNode* pb) {
    l_OperationBase* operation_base = malloc(sizeof(l_OperationBase));
    operation_base->pa = pa;
    operation_base->pb = pb;
    l_ComputeNode* addition = malloc(sizeof(l_ComputeNode));
    addition->evaluate_func_p = div_evaluate;
    addition->requires_grad_func_p = operation_requires_grad;
    addition->base = (void*)operation_base;
    return addition;
}

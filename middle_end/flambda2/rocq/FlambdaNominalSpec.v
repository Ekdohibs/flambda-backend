(******************************************************************************
 *                                  OxCaml                                    *
 * -------------------------------------------------------------------------- *
 *                               MIT License                                  *
 *                                                                            *
 * Copyright (c) 2026 Jane Street Group LLC                                   *
 * opensource-contacts@janestreet.com                                         *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included    *
 * in all copies or substantial portions of the Software.                     *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER *
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    *
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        *
 * DEALINGS IN THE SOFTWARE.                                                  *
 ******************************************************************************)

(* Names: renamings, free names, and the binders of Flambda 2 as name
   abstractions.

   This file is to be reviewed, against middle_end/flambda2/nominal/: the
   renamings and their application to terms ([Renaming], [apply_renaming]),
   the statement about the binders of code, and the three about
   bindings that are not used, with the relations of free names that
   they are written with. Those relations are an older account, kept
   for what is listed where they are defined; the free names
   ([free_names]) are in FlambdaBindersSpec.v.

   The compiler treats a binder as a name abstraction in the sense of
   nominal sets: [Name_abstraction.t] holds a bound name and a term,
   [apply_renaming] acts on it by a permutation of names, and its only
   eliminator, [pattern_match], gives the term with the bound name replaced
   by a fresh one. So a term stands for its alpha-equivalence class. Rocq
   has no quotients, and the terms of FlambdaModel.v stay what they are,
   expressions with named binders that shadow; the semantics needs nothing
   else. What this file adds is the part of the theory of nominal sets that
   the compiler uses, with [expr_equivalent] in the place of equality of
   classes:
   - the permutations of names with finite support, as lists of swaps, and
     a renaming as one permutation of variables and one of continuations;
   - their action on terms ([rename_expr]), which renames bound and free
     occurrences alike and so never captures. It goes into code, as the
     compiler's does: the parameters of code, [my_closure], its two
     continuations and its body are renamed;
   - the support of a term up to its binders, its free names: they are
     the functions of FlambdaBindersSpec.v, one for each sort, which go
     into code;
   - the defining property of an abstraction: a permutation that fixes the
     free names of a binder with its scope may be applied to the bound name
     and the scope. For a [Let], a [Let_cont] and a handler the
     statements are in FlambdaCanonicalFreeNamesSpec.v
     ([rename_let_canonical_statement] and the four after it); the
     concretion at a fresh name that [pattern_match] performs is the
     case of a swap ([freshen_let_canonical_statement]). The binders of
     code are the case with no condition on the permutation: closed code
     has no free name ([rename_code_binders_statement], here);
   - that behaviour depends on the free names only, by which a binding
     that is not used may be removed ([unused_let_statement] and the two
     after it, here), a [Let] of a primitive of the heap excepted: what
     that does is seen in the heap of an exit.
   With FlambdaContextSpec.v these apply to any subterm.

   Each statement that renames takes as a premise that the code bound in
   what is renamed is closed (FlambdaBindersSpec.code_closed, a
   condition of well-formedness). No statement says anything of a
   renaming of code that is not closed.

   The relations [var_free_in_expr] and [cont_free_in_expr] of this
   file are an older account of the free names, which does not go into
   code. The statements that a bound name may be renamed were written
   with them, in this file; they are now written with the functions,
   and the older forms are lemmas of FlambdaNominalProofs.v, under the
   names they had, because proofs use them. The three statements about
   a binding that is not used are still written with the relations,
   and are meant to be: the comment over them says why.
   FlambdaFreeNamesSpec.v says that the two accounts agree where code
   is closed.

   There is no type of alpha-equivalence classes here. The general
   notions (a type whose values contain names, its support, name
   abstractions and their laws) are in FlambdaNameAbstractionSpec.v, and
   the equivariance of the machine in FlambdaEquivarianceSpec.v. Fresh
   names are not generated here; a statement holds of every name that is
   fresh enough.

   The proofs are in FlambdaNominalProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaBindersSpec FlambdaHeapSpec FlambdaSemanticsSpec.
Import ListNotations.

(* ======================================================================== *)
(* Renamings                                                                *)
(* ======================================================================== *)

(* A permutation of names: the swaps of the list, the last one first. *)
Definition permutation : Type := list (positive * positive).

Definition swap (a b name : positive) : positive :=
  if Pos.eqb name a then b else if Pos.eqb name b then a else name.

Fixpoint permute (p : permutation) (name : positive) : positive :=
  match p with
  | [] => name
  | (a, b) :: p => swap a b (permute p name)
  end.

(* [Renaming.t], for variables and continuations. *)
Record renaming : Type := Renaming {
  variable_permutation : permutation;
  continuation_permutation : permutation
}.

Definition identity_renaming : renaming := Renaming [] [].

Definition rename_variable (r : renaming) (x : variable) : variable :=
  permute (variable_permutation r) x.

Definition rename_continuation (r : renaming) (k : continuation)
    : continuation :=
  permute (continuation_permutation r) k.

Arguments rename_variable : simpl never.
Arguments rename_continuation : simpl never.

(* [apply_renaming], on each sort of term. *)
Definition rename_simple (r : renaming) (s : simple) : simple :=
  match s with
  | Var x => Var (rename_variable r x)
  | Symbol sym => Symbol sym
  | Const c => Const c
  end.

Definition rename_named (r : renaming) (defining_expr : named) : named :=
  match defining_expr with
  | Simple s => Simple (rename_simple r s)
  | Prim p args => Prim p (map (rename_simple r) args)
  | Set_of_closures code value_slots =>
    Set_of_closures code
      (map (fun slot => (fst slot, rename_simple r (snd slot))) value_slots)
  | Closure_of_set slot function_decls value_slots =>
    Closure_of_set slot function_decls
      (map (fun slot => (fst slot, rename_simple r (snd slot))) value_slots)
  | Project_value_slot slot closure =>
    Project_value_slot slot (rename_simple r closure)
  | Project_value_slot_from project_from slot closure =>
    Project_value_slot_from project_from slot (rename_simple r closure)
  | Project_function_slot move_from move_to closure =>
    Project_function_slot move_from move_to (rename_simple r closure)
  end.

Definition rename_trap_action (r : renaming) (action : trap_action)
    : trap_action :=
  match action with
  | Push exn_handler => Push (rename_continuation r exn_handler)
  | Pop exn_handler kind => Pop (rename_continuation r exn_handler) kind
  end.

Definition rename_apply_cont (r : renaming) (ac : apply_cont) : apply_cont :=
  Apply_cont_expr (rename_continuation r (ac_continuation ac))
    (map (rename_simple r) (ac_args ac))
    (option_map (rename_trap_action r) (ac_trap_action ac)).

(* A renaming acts on the variables and the continuations of a term.
   Symbols, code ids, value slots and function slots are not renamed.
   Kinds, arities and metadata have no names in them and are left as
   they are. *)
Definition rename_static_const (r : renaming) (const : static_const)
    : static_const :=
  match const with
  | Static_const_set_of_closures code value_slots =>
    Static_const_set_of_closures code
      (map (fun slot => (fst slot, rename_simple r (snd slot))) value_slots)
  | Static_const_closure_of_set slot closure_symbols function_decls
      value_slots =>
    Static_const_closure_of_set slot closure_symbols function_decls
      (map (fun slot => (fst slot, rename_simple r (snd slot))) value_slots)
  | Static_const_block tag fields =>
    Static_const_block tag (map (rename_simple r) fields)
  | Static_const_boxed_number contents =>
    Static_const_boxed_number (rename_simple r contents)
  | Static_const_immutable_string bytes => Static_const_immutable_string bytes
  | Static_const_immutable_value_array elements =>
    Static_const_immutable_value_array (map (rename_simple r) elements)
  end.

Definition rename_apply (r : renaming) (ap : apply_expr) : apply_expr :=
  Apply_expr (rename_simple r (ap_callee ap)) (ap_call_kind ap)
    (map (rename_simple r) (ap_args ap))
    (rename_continuation r (ap_continuation ap))
    (rename_continuation r (ap_exn_continuation ap))
    (map (fun arg => (rename_simple r (fst arg), snd arg))
       (ap_exn_extra_args ap))
    (ap_args_arity ap) (ap_return_arity ap).

(* [apply_renaming], [apply_renaming_continuation_handler],
   [apply_renaming_static_const_or_code] and
   [apply_renaming_function_params_and_body] of terms/flambda.ml. Code is
   renamed as everything else is: its parameters, [my_closure], its two
   continuations and its body, as the compiler renames it
   ([Code0.apply_renaming]). *)
Fixpoint rename_expr (r : renaming) (e : expr) : expr :=
  match e with
  | Let x kind defining_expr body =>
    Let (rename_variable r x) kind (rename_named r defining_expr)
      (rename_expr r body)
  | Let_static bindings body =>
    Let_static (map (rename_static_binding r) bindings) (rename_expr r body)
  | Let_cont_non_recursive k handler body =>
    Let_cont_non_recursive (rename_continuation r k)
      (rename_handler r handler) (rename_expr r body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive
      (map (fun '(k, h) => (rename_continuation r k, rename_handler r h))
         handlers)
      (rename_expr r body)
  | Apply ap => Apply (rename_apply r ap)
  | Apply_cont ac => Apply_cont (rename_apply_cont r ac)
  | Switch scrutinee arms =>
    Switch (rename_simple r scrutinee)
      (map (fun arm => (fst arm, rename_apply_cont r (snd arm))) arms)
  | Invalid => Invalid
  end
with rename_handler (r : renaming) (h : continuation_handler)
    : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler (map (rename_variable r) params) metadata (rename_expr r handler)
  end
with rename_static_binding (r : renaming) (binding : static_binding)
    : static_binding :=
  match binding with
  | Bind_code id c => Bind_code id (rename_code r c)
  | Bind_deleted_code id => Bind_deleted_code id
  | Bind_symbol s const => Bind_symbol s (rename_static_const r const)
  end
with rename_code (r : renaming) (c : code) : code :=
  match c with
  | Code params my_closure return_continuation exn_continuation metadata
      body =>
    Code (map (rename_variable r) params) (rename_variable r my_closure)
      (rename_continuation r return_continuation)
      (rename_continuation r exn_continuation) metadata (rename_expr r body)
  end.

Definition rename_handlers (r : renaming)
    (handlers : list (continuation * continuation_handler))
    : list (continuation * continuation_handler) :=
  map (fun '(k, h) => (rename_continuation r k, rename_handler r h)) handlers.

(* ======================================================================== *)
(* The older relations of free names                                        *)
(* ======================================================================== *)

(* A variable, or a continuation, is free in a term. These relations do
   not go into code: a name that occurs only in the code of a term is
   not free by them. The free names of a term are the functions of
   FlambdaBindersSpec.v, which do, and no statement that a bound name
   may be renamed, and no rule of FlambdaAlphaSpec.alpha_equivalent, is
   written with these relations. They are still here for:
   - the three statements at the end of this file about a binding that
     is not used, which ask nothing of the code;
   - FlambdaApplySpec.inlinable, of code whose body binds nothing
     static, so that there is no code in it for the two accounts to
     differ on;
   - one statement of FlambdaRenameApartSpec.v;
   - FlambdaFreeNamesSpec.v, which says how they relate to the
     functions;
   - the proofs, many of which are written with them, and the tests of
     freedom that the rewrites of the validator run
     (FlambdaNormaliseSpec.var_free_b, FlambdaUnusedContsSpec.cont_free_b),
     which do not go into code either and are proved sound for these.
   [var_free_in_simple] and [var_free_in_simples] say that a variable is
   one of some simples, where there is no code and no binder. *)

(* Some handler of the list satisfies [P]. *)
Fixpoint any_handler {A : Type} (P : A -> Prop)
    (handlers : list (continuation * A)) : Prop :=
  match handlers with
  | [] => False
  | (_, h) :: handlers => P h ∨ any_handler P handlers
  end.

Definition var_free_in_simple (x : variable) (s : simple) : Prop :=
  match s with
  | Var y => x = y
  | Symbol _ | Const _ => False
  end.

Definition var_free_in_simples (x : variable) (args : list simple) : Prop :=
  Exists (var_free_in_simple x) args.

Definition var_free_in_named (x : variable) (defining_expr : named) : Prop :=
  match defining_expr with
  | Simple s => var_free_in_simple x s
  | Prim _ args => var_free_in_simples x args
  | Set_of_closures _ value_slots | Closure_of_set _ _ value_slots =>
    Exists (fun slot => var_free_in_simple x (snd slot)) value_slots
  | Project_value_slot _ closure | Project_value_slot_from _ _ closure
  | Project_function_slot _ _ closure =>
    var_free_in_simple x closure
  end.

Definition var_free_in_static_const (x : variable) (const : static_const)
    : Prop :=
  match const with
  | Static_const_set_of_closures _ value_slots
  | Static_const_closure_of_set _ _ _ value_slots =>
    Exists (fun slot => var_free_in_simple x (snd slot)) value_slots
  | Static_const_block _ fields => var_free_in_simples x fields
  | Static_const_immutable_value_array elements =>
    var_free_in_simples x elements
  | Static_const_boxed_number contents => var_free_in_simple x contents
  | Static_const_immutable_string _ => False
  end.

(* Code has no free variable of the term around it, whatever is in it. *)
Definition var_free_in_static_binding (x : variable)
    (binding : static_binding) : Prop :=
  match binding with
  | Bind_code _ _ | Bind_deleted_code _ => False
  | Bind_symbol _ const => var_free_in_static_const x const
  end.

(* A [Let] binds its variable in its body; a handler binds its
   parameters. *)
Fixpoint var_free_in_expr (x : variable) (e : expr) : Prop :=
  match e with
  | Let y _ defining_expr body =>
    var_free_in_named x defining_expr ∨ (x <> y ∧ var_free_in_expr x body)
  | Let_static bindings body =>
    Exists (var_free_in_static_binding x) bindings ∨ var_free_in_expr x body
  | Let_cont_non_recursive _ handler body =>
    var_free_in_handler x handler ∨ var_free_in_expr x body
  | Let_cont_recursive handlers body =>
    any_handler (var_free_in_handler x) handlers ∨ var_free_in_expr x body
  | Apply ap =>
    var_free_in_simple x (ap_callee ap) ∨ var_free_in_simples x (ap_args ap)
    ∨ var_free_in_simples x (exn_extra_simples ap)
  | Apply_cont ac => var_free_in_simples x (ac_args ac)
  | Switch scrutinee arms =>
    var_free_in_simple x scrutinee
    ∨ Exists (fun arm => var_free_in_simples x (ac_args (snd arm))) arms
  | Invalid => False
  end
with var_free_in_handler (x : variable) (h : continuation_handler) : Prop :=
  match h with
  | Handler params _ handler => x ∉ params ∧ var_free_in_expr x handler
  end.

(* A non-recursive [Let_cont] binds its continuation in its body; a
   recursive one binds its continuations in its body and its handlers. The
   continuation that a trap action names is free as the one applied is
   ([Name_occurrences] counts the two apart). *)
Fixpoint cont_free_in_expr (k : continuation) (e : expr) : Prop :=
  match e with
  | Let _ _ _ body | Let_static _ body => cont_free_in_expr k body
  | Let_cont_non_recursive bound handler body =>
    cont_free_in_handler k handler ∨ (k <> bound ∧ cont_free_in_expr k body)
  | Let_cont_recursive handlers body =>
    lookup k handlers = None
    ∧ (any_handler (cont_free_in_handler k) handlers
       ∨ cont_free_in_expr k body)
  | Apply ap => k = ap_continuation ap ∨ k = ap_exn_continuation ap
  | Apply_cont ac => k ∈ apply_cont_continuations ac
  | Switch _ arms =>
    Exists (fun arm => k ∈ apply_cont_continuations (snd arm)) arms
  | Invalid => False
  end
with cont_free_in_handler (k : continuation) (h : continuation_handler)
    : Prop :=
  match h with
  | Handler _ _ handler => cont_free_in_expr k handler
  end.

(* ======================================================================== *)
(* Binders are name abstractions                                            *)
(* ======================================================================== *)

(* The binders of a [Let], of a [Let_cont] and of a handler: that a
   renaming which fixes the free names of the scope may be applied to
   the bound names and the scope is said in
   FlambdaCanonicalFreeNamesSpec.v, with the free names of
   FlambdaBindersSpec.v. *)

(* The binders of code: its parameters, [my_closure] and its return and
   exception continuations. Closed code has no other name, so any
   renaming at all may be applied to it. *)
Definition rename_code_binders_statement : Prop :=
  ∀ (r : renaming) (before : list static_binding) (id : code_id) (c : code)
    (after : list static_binding) (body : expr),
    code_is_closed c = true ->
    expr_equivalent (Let_static (before ++ Bind_code id c :: after) body)
      (Let_static (before ++ Bind_code id (rename_code r c) :: after) body).

(* ======================================================================== *)
(* Bindings that are not used                                               *)
(* ======================================================================== *)

(* These three are written with the older relations, on purpose. They
   ask nothing of the code, and the relations do not go into code: a
   name that occurs only in code that does not bind it is not free by
   them, and the binding may still be removed, since the machine runs
   code under its parameters and [my_closure] alone and code never
   reads such a name. The rewrites of the validator that remove
   bindings test freedom in the same way
   (FlambdaNormaliseSpec.var_free_b, FlambdaUnusedContsSpec.cont_free_b),
   are stated of every term, and are proved from these. With the free
   names of FlambdaBindersSpec.v the premise would ask more, and the
   statements would say less of a term whose code is not closed; those
   forms are corollaries, in FlambdaCanonicalFreeNamesSpec.v. *)

(* A [Let] whose variable is not free in its body may be removed, if
   its defining expression is not a primitive of the heap
   (FlambdaHeapSpec.named_changes_heap). This is a refinement: the
   defining expression may be undefined. A primitive of the heap is
   not removed: it changes the heap, and an exit is compared with its
   heap. Simplify removes more, an unused [Let] of an allocation
   among them ([Named.at_most_generative_effects]); that is not a
   refinement in the sense of this file for as long as two exits are
   compared with their heaps equal (decision D40). *)
Definition unused_let_statement : Prop :=
  ∀ (x : variable) (kind : flambda_kind) (defining_expr : named)
    (body : expr),
    named_changes_heap defining_expr = false ->
    ~ var_free_in_expr x body ->
    expr_refines (Let x kind defining_expr body) body.

Definition unused_let_cont_statement : Prop :=
  ∀ (k : continuation) (handler : continuation_handler) (body : expr),
    ~ cont_free_in_expr k body ->
    expr_equivalent (Let_cont_non_recursive k handler body) body.

Definition unused_let_cont_recursive_statement : Prop :=
  ∀ (handlers : list (continuation * continuation_handler)) (body : expr),
    handlers_bind_statics handlers = false ->
    (∀ k : continuation,
      cont_free_in_expr k body -> lookup k handlers = None) ->
    expr_equivalent (Let_cont_recursive handlers body) body.

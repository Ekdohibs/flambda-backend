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

(* Giving every binder of a term a name of its own: a normalisation that
   is an equivalence.

   This file is to be reviewed: [rename_apart_avoiding] and
   [rename_apart], which are extracted and run by the validation of
   middle_end/flambda2/tests/semantics/; the names of a term
   ([variables_of], [continuations_of]), which one statement of
   FlambdaRenameApartCanonicalSpec.v mentions; and the statements. What
   makes the candidate ([canonical_avoiding] and what it is made of)
   need not be read for the first statement: [rename_apart_avoiding]
   keeps the candidate only if the comparison up to bound names
   (FlambdaCompareSpec.expr_eqb) says it is the term, and gives the term
   back as it was otherwise. So that statement holds whatever the
   candidate is, by the soundness of the comparison alone.

   Why it is needed. Simplify gives fresh names to the binders of a
   function body each time it inlines it. The inlining of the validation
   (FlambdaNormaliseSpec.inline_calls) copies the body as it is, so a
   function inlined twice leaves two [Let]s of the same variable, one in
   the scope of the other:

     let v = x * x in let r = v in let v = y * y in ... v + r ...

   FlambdaAliasesSpec.follow_aliases forgets that [r] is [v] where [v] is
   bound again, as it must, the alias stays, and the term is not the one
   Simplify made. With every binder renamed apart nothing is bound twice
   and the alias is followed.

   The candidate. Every binder of the term is given a new name, and its
   uses are renamed: the variable of a [Let], the parameters of a
   handler, the continuation of a [Let_cont], and the binders of code
   (its parameters, its closure, its two continuations). Symbols, code
   ids and slots are left alone. There are as many new names as the
   term has binders ([binders]), and they are made by
   FlambdaFreshQuantifierSpec.fresh_names, different from one another,
   from every free name of the term, those of its code included
   (FlambdaBindersSpec.free_variables_of_expr,
   free_continuations_of_expr), and from the continuations that the
   caller says to avoid; the term is gone through once, each binder
   taking the next of them ([apart]). A new name may be a name that the
   term binds: the new names are made from the free names alone, so two
   terms that differ in their bound names only have the same candidate,
   which is a canonical form of the term ([canonical_avoiding];
   FlambdaRenameApartCanonicalSpec.v says what it is). The candidate
   that this file had first, [renamed_apart_avoiding], keeps its new
   names away from every name of the term, bound names too; nothing
   runs it now, and FlambdaRenameApartAlphaSpec.v and
   FlambdaRenameApartCanonicalSpec.v have statements about it. The uses
   are renamed by the renamings of FlambdaNominalSpec.v, which are
   permutations: at a binder, what the name is under the renaming so far
   is swapped with the new name ([bind_variable], [bind_continuation]).
   This file makes no name by itself and does not look at the numbers of
   names.

   Renaming. [rename_apart] does not commute with a renaming of the
   term: the new names are made from the free names of the term, and
   those of the renamed term are others. It commutes up to the new
   names, on a term whose code is closed: for a renaming [r] there is a
   renaming that does to the free variables and continuations of the
   term, those of its code included, what [r] does, and with which the
   result for the renamed term is the renamed result. The statement is
   in FlambdaRenameApartCanonicalSpec.v, with what it is proved from
   ([rename_apart_equivariant_up_to_statement]). The form in which the
   second renaming does what [r] does to every name of the term, bound
   names too, which held of the first candidate, is false of this one
   (FlambdaRenameApartCanonicalSpec
   .canonical_avoiding_not_equivariant_up_to_names_statement).

   The continuations to avoid. A term may stand where continuations are
   in scope that it does not mention: those of the unit it is part of.
   A new name must not be one of them if the term is later put where
   they are used. [rename_apart_avoiding] is given such continuations
   and makes no new name that is one of them; [rename_apart] is given
   none. Under a renaming the continuations to avoid are renamed with
   the term, and the renaming that the statement gives agrees with [r]
   on them too (FlambdaRenameApartCanonicalSpec
   .rename_apart_avoiding_equivariant_up_to_statement).
   The numbers of the continuations to avoid are avoided for the new
   variables as well, which is more than is needed and harms nothing.

   The proofs are in FlambdaRenameApartProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaBindersSpec FlambdaCompareSpec
  FlambdaNominalSpec FlambdaNameAbstractionSpec FlambdaFreshQuantifierSpec.
Import ListNotations.

(* ======================================================================== *)
(* The names of a term                                                      *)
(* ======================================================================== *)

Definition variables_in_simple (s : simple) : list variable :=
  match s with
  | Var x => [x]
  | Symbol _ | Const _ => []
  end.

Definition variables_in_simples (args : list simple) : list variable :=
  flat_map variables_in_simple args.

Definition variables_in_slots (slots : list (value_slot * simple))
    : list variable :=
  flat_map (fun slot => variables_in_simple (snd slot)) slots.

Definition variables_in_named (d : named) : list variable :=
  match d with
  | Simple s => variables_in_simple s
  | Prim _ args => variables_in_simples args
  | Set_of_closures _ slots | Closure_of_set _ _ slots =>
    variables_in_slots slots
  | Project_value_slot _ closure | Project_value_slot_from _ _ closure
  | Project_function_slot _ _ closure =>
    variables_in_simple closure
  end.

Definition variables_in_static_const (const : static_const)
    : list variable :=
  match const with
  | Static_const_set_of_closures _ slots
  | Static_const_closure_of_set _ _ _ slots =>
    variables_in_slots slots
  | Static_const_block _ fields => variables_in_simples fields
  | Static_const_immutable_value_array elements =>
    variables_in_simples elements
  | Static_const_boxed_number contents => variables_in_simple contents
  | Static_const_immutable_string _ => []
  end.

Definition variables_in_apply_cont (ac : apply_cont) : list variable :=
  variables_in_simples (ac_args ac).

Definition variables_in_apply (ap : apply_expr) : list variable :=
  variables_in_simple (ap_callee ap) ++ variables_in_simples (ap_args ap)
  ++ variables_in_simples (exn_extra_simples ap).

(* The variables of an expression, bound or not, in its code too. *)
Fixpoint variables_of (e : expr) : list variable :=
  match e with
  | Let x _ d body => x :: variables_in_named d ++ variables_of body
  | Let_static bindings body =>
    flat_map variables_in_binding bindings ++ variables_of body
  | Let_cont_non_recursive _ h body =>
    variables_in_handler h ++ variables_of body
  | Let_cont_recursive handlers body =>
    flat_map (fun kh => variables_in_handler (snd kh)) handlers
    ++ variables_of body
  | Apply ap => variables_in_apply ap
  | Apply_cont ac => variables_in_apply_cont ac
  | Switch scrutinee arms =>
    variables_in_simple scrutinee
    ++ flat_map (fun arm => variables_in_apply_cont (snd arm)) arms
  | Invalid => []
  end
with variables_in_handler (h : continuation_handler) : list variable :=
  match h with
  | Handler params _ handler => params ++ variables_of handler
  end
with variables_in_binding (binding : static_binding) : list variable :=
  match binding with
  | Bind_code _ c => variables_in_code c
  | Bind_deleted_code _ => []
  | Bind_symbol _ const => variables_in_static_const const
  end
with variables_in_code (c : code) : list variable :=
  match c with
  | Code params my_closure _ _ _ body =>
    my_closure :: params ++ variables_of body
  end.

Definition continuations_in_apply (ap : apply_expr) : list continuation :=
  [ap_continuation ap; ap_exn_continuation ap].

(* Its continuations, likewise. *)
Fixpoint continuations_of (e : expr) : list continuation :=
  match e with
  | Let _ _ _ body => continuations_of body
  | Let_static bindings body =>
    flat_map continuations_in_binding bindings ++ continuations_of body
  | Let_cont_non_recursive k h body =>
    k :: continuations_in_handler h ++ continuations_of body
  | Let_cont_recursive handlers body =>
    flat_map (fun kh => fst kh :: continuations_in_handler (snd kh)) handlers
    ++ continuations_of body
  | Apply ap => continuations_in_apply ap
  | Apply_cont ac => apply_cont_continuations ac
  | Switch _ arms =>
    flat_map (fun arm => apply_cont_continuations (snd arm)) arms
  | Invalid => []
  end
with continuations_in_handler (h : continuation_handler)
    : list continuation :=
  match h with
  | Handler _ _ handler => continuations_of handler
  end
with continuations_in_binding (binding : static_binding)
    : list continuation :=
  match binding with
  | Bind_code _ c => continuations_in_code c
  | Bind_deleted_code _ | Bind_symbol _ _ => []
  end
with continuations_in_code (c : code) : list continuation :=
  match c with
  | Code _ _ return_continuation exn_continuation _ body =>
    return_continuation :: exn_continuation :: continuations_of body
  end.

(* The names of an expression, of both sorts. *)
Definition names_of (e : expr) : list positive :=
  variables_of e ++ continuations_of e.

(* ======================================================================== *)
(* The candidate                                                            *)
(* ======================================================================== *)

(* How many binders an expression has: the new names it takes. *)
Fixpoint binders (e : expr) : nat :=
  match e with
  | Let _ _ _ body => S (binders body)
  | Let_static bindings body =>
    list_sum (map binders_in_binding bindings) + binders body
  | Let_cont_non_recursive _ h body => binders_in_handler h + S (binders body)
  | Let_cont_recursive handlers body =>
    length handlers
    + list_sum (map (fun kh => binders_in_handler (snd kh)) handlers)
    + binders body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => O
  end
with binders_in_handler (h : continuation_handler) : nat :=
  match h with
  | Handler params _ handler => length params + binders handler
  end
with binders_in_binding (binding : static_binding) : nat :=
  match binding with
  | Bind_code _ c => binders_in_code c
  | Bind_deleted_code _ | Bind_symbol _ _ => O
  end
with binders_in_code (c : code) : nat :=
  match c with
  | Code params _ _ _ _ body => S (length params + S (S (binders body)))
  end.

(* The renaming under which the variable, or the continuation, has the
   new name as well: what the name is under the renaming is swapped with
   the new name. *)
Definition bind_variable (r : renaming) (x x' : variable) : renaming :=
  Renaming ((rename_variable r x, x') :: variable_permutation r)
    (continuation_permutation r).

Definition bind_continuation (r : renaming) (k k' : continuation)
    : renaming :=
  Renaming (variable_permutation r)
    ((rename_continuation r k, k') :: continuation_permutation r).

(* The next new name for a variable, and for a continuation: the first
   of those left, and the name as it is renamed already where none is
   left. *)
Definition new_variable (r : renaming) (news : list positive) (x : variable)
    : variable :=
  hd (rename_variable r x) news.

Definition new_continuation (r : renaming) (news : list positive)
    (k : continuation) : continuation :=
  hd (rename_continuation r k) news.

(* New names for the variables given, bound one after the other: the new
   names left, those taken, and the renaming with them bound. *)
Fixpoint bind_variables (r : renaming) (news : list positive)
    (xs : list variable) : list positive * list variable * renaming :=
  match xs with
  | [] => (news, [], r)
  | x :: xs =>
    let x' := new_variable r news x in
    let '(rest, xs', r') :=
      bind_variables (bind_variable r x x') (tl news) xs
    in
    (rest, x' :: xs', r')
  end.

Fixpoint bind_continuations (r : renaming) (news : list positive)
    (ks : list continuation)
    : list positive * list continuation * renaming :=
  match ks with
  | [] => (news, [], r)
  | k :: ks =>
    let k' := new_continuation r news k in
    let '(rest, ks', r') :=
      bind_continuations (bind_continuation r k k') (tl news) ks
    in
    (rest, k' :: ks', r')
  end.

(* A function that takes new names applied to each element of a list in
   turn: the new names left and the results. *)
Fixpoint thread {A B : Type} (f : list positive -> A -> list positive * B)
    (news : list positive) (l : list A) : list positive * list B :=
  match l with
  | [] => (news, [])
  | a :: l =>
    let '(rest, b) := f news a in
    let '(rest', bs) := thread f rest l in
    (rest', b :: bs)
  end.

(* An expression with its binders renamed, under the renaming of the
   variables and the continuations in scope: the new names left and the
   expression. *)
Fixpoint apart (r : renaming) (news : list positive) (e : expr) {struct e}
    : list positive * expr :=
  match e with
  | Let x kind d body =>
    let x' := new_variable r news x in
    let '(rest, body') := apart (bind_variable r x x') (tl news) body in
    (rest, Let x' kind (rename_named r d) body')
  | Let_static bindings body =>
    let '(rest, bindings') := thread (apart_binding r) news bindings in
    let '(rest', body') := apart r rest body in
    (rest', Let_static bindings' body')
  | Let_cont_non_recursive k h body =>
    let '(rest, h') := apart_handler r news h in
    let k' := new_continuation r rest k in
    let '(rest', body') := apart (bind_continuation r k k') (tl rest) body in
    (rest', Let_cont_non_recursive k' h' body')
  | Let_cont_recursive handlers body =>
    let '(rest, ks', r') := bind_continuations r news (map fst handlers) in
    let '(rest', hs') :=
      thread (fun news kh => apart_handler r' news (snd kh)) rest handlers
    in
    let '(rest'', body') := apart r' rest' body in
    (rest'', Let_cont_recursive (combine ks' hs') body')
  | Apply ap => (news, Apply (rename_apply r ap))
  | Apply_cont ac => (news, Apply_cont (rename_apply_cont r ac))
  | Switch scrutinee arms =>
    (news,
     Switch (rename_simple r scrutinee)
       (map (fun arm => (fst arm, rename_apply_cont r (snd arm))) arms))
  | Invalid => (news, Invalid)
  end
with apart_handler (r : renaming) (news : list positive)
    (h : continuation_handler) {struct h}
    : list positive * continuation_handler :=
  match h with
  | Handler params metadata handler =>
    let '(rest, params', r') := bind_variables r news params in
    let '(rest', handler') := apart r' rest handler in
    (rest', Handler params' metadata handler')
  end
with apart_binding (r : renaming) (news : list positive)
    (binding : static_binding) {struct binding}
    : list positive * static_binding :=
  match binding with
  | Bind_code id c =>
    let '(rest, c') := apart_code r news c in
    (rest, Bind_code id c')
  | Bind_deleted_code _ => (news, binding)
  | Bind_symbol s const => (news, Bind_symbol s (rename_static_const r const))
  end
with apart_code (r : renaming) (news : list positive) (c : code) {struct c}
    : list positive * code :=
  match c with
  | Code params my_closure return_continuation exn_continuation metadata
      body =>
    let my_closure' := new_variable r news my_closure in
    let r := bind_variable r my_closure my_closure' in
    let '(rest, params', r) := bind_variables r (tl news) params in
    let return_continuation' := new_continuation r rest return_continuation in
    let r := bind_continuation r return_continuation return_continuation' in
    let exn_continuation' := new_continuation r (tl rest) exn_continuation in
    let r := bind_continuation r exn_continuation exn_continuation' in
    let '(rest', body') := apart r (tl (tl rest)) body in
    (rest',
     Code params' my_closure' return_continuation' exn_continuation' metadata
       body')
  end.

(* The first candidate, which nothing runs now: the binders of the
   expression given new names that are different from one another, from
   every name of the expression and from the continuations [avoid]. *)
Definition renamed_apart_avoiding (avoid : list continuation) (e : expr)
    : expr :=
  snd
    (apart identity_renaming
       (fresh_names (binders e) (avoid ++ names_of e)) e).

(* The candidate: the binders of the expression given new names that
   are different from one another, from every free name of the
   expression and from the continuations [avoid]. A new name may be a
   name that the expression binds. *)
Definition canonical_avoiding (avoid : list continuation) (e : expr)
    : expr :=
  snd
    (apart identity_renaming
       (fresh_names (binders e)
          (avoid ++ free_variables_of_expr e ++ free_continuations_of_expr e))
       e).

(* ======================================================================== *)
(* The normalisation                                                        *)
(* ======================================================================== *)

(* The candidate if it is the term up to bound names, and the term if
   not. *)
Definition rename_apart_avoiding (avoid : list continuation) (e : expr)
    : expr :=
  let e' := canonical_avoiding avoid e in
  if expr_eqb [] [] e e' then e' else e.

(* With no continuation to avoid. *)
Definition rename_apart (e : expr) : expr := rename_apart_avoiding [] e.

Definition rename_apart_avoiding_equivalent_statement : Prop :=
  ∀ (avoid : list continuation) (e : expr),
    expr_equivalent e (rename_apart_avoiding avoid e).

Definition rename_apart_equivalent_statement : Prop :=
  ∀ e : expr, expr_equivalent e (rename_apart e).

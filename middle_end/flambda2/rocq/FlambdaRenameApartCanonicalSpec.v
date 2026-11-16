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

(* A canonical form for the bound names of a term.

   This file is to be reviewed: the definitions and the statements.

   FlambdaRenameApartSpec.renamed_apart_avoiding gives every binder of
   a term a new name, and takes the new names above every name of the
   term, bound names included. So two terms that differ in a bound
   name can get different new names. [canonical_avoiding] is the same
   function with the new names taken above the free names of the term
   only (and above the continuations [avoid]): the supply is the same
   for two terms that have the same free names and as many binders.

   [canonical_avoiding] is the candidate, with no check of it at each
   run (FlambdaRenameApartSpec.rename_apart_avoiding has one).

   [bound_names] lists the names that the binders of a term bind, and
   [binders_apart] says that they are different from one another and
   from the names of a list.

   - [alpha_equivalent_canonical_statement]: alpha-equivalent terms
     (FlambdaAlphaSpec.alpha_equivalent) have the same canonical form.
     It is an equality of terms and has no premise. So a function
     applied to the canonical form gives equal results for
     alpha-equivalent terms, whatever the function, with the same
     [avoid] on both sides; it says nothing of what the function
     gives being canonical.
   - [renamed_apart_avoiding_bound_names_statement],
     [canonical_avoiding_bound_names_statement]: the names that each
     candidate binds are the new names, in the order of [bound_names].
     No premise.
   - [renamed_apart_avoiding_binders_apart_statement],
     [canonical_avoiding_binders_apart_statement]: so the binders of
     each candidate are different from one another, from the
     continuations to avoid and from what the new names were taken
     above: every name of the term for the first, its free names for
     the second. No premise.
   - [canonical_avoiding_alpha_statement]: the canonical form of a term
     whose code is closed and whose binders are distinct is
     alpha-equivalent to the term. The two premises are those of the
     same statement for [renamed_apart_avoiding]
     (FlambdaRenameApartAlphaSpec.v), where a term shows that distinct
     binders are needed.
   - [alpha_equivalent_iff_canonical_statement]: so two such terms are
     alpha-equivalent exactly when their canonical forms are equal.
   - [canonical_avoiding_idempotent_statement]: and the canonical form
     of such a term is its own canonical form, which
     [renamed_apart_avoiding] is not: its new names are above those
     that the term binds, so they go up at each call.
   - [canonical_avoiding_equivalent_statement],
     [canonical_avoiding_accepted_statement]: the canonical form of
     such a term behaves as the term does, and the comparison up to
     bound names accepts it.
   - [renamed_apart_avoiding_free_names_statement],
     [canonical_avoiding_free_names_statement]: the free names of each
     candidate are among those of the term. No premise.
   - [canonical_avoiding_equivariant_up_to_statement]: the canonical
     form of a renamed term is the canonical form of the term under a
     second renaming, which does to the free names of that form, and
     to the continuations to avoid, what the first does. No premise.
     It is FlambdaRenameApartSpec
     .rename_apart_avoiding_equivariant_up_to_statement for this
     candidate. The second renaming cannot do to the names that the
     term binds what the first does, as the one of
     FlambdaRenameApartSpec
     .rename_apart_equivariant_up_to_names_statement does: here a new
     name may be a name that the term binds.
   - [canonical_avoiding_not_equivariant_up_to_names_statement]: so
     the form of that statement, whose second renaming does to every
     name of the term what the first does, is false of this candidate.
     The proof has the term: [let x = 0 in cont k (x)] with [x] the
     first name above [k] is its own canonical form, and the canonical
     form of the term with [x] renamed.
   - [renamed_apart_avoiding_code_closed_statement],
     [canonical_avoiding_code_closed_statement]: each candidate of a
     term whose code is closed has its code closed. No premise on the
     binders: a candidate that is not the term up to bound names has
     closed code all the same.

   Nothing is trusted here. The proofs are in
   FlambdaRenameApartCanonicalProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaBindersSpec FlambdaCompareSpec
  FlambdaNominalSpec FlambdaAlphaSpec FlambdaFreshQuantifierSpec
  FlambdaRenameApartSpec.
Import ListNotations.

(* ======================================================================== *)
(* The canonical form                                                       *)
(* ======================================================================== *)

(* The binders of the expression given new names that are different
   from one another, from every free name of the expression and from
   the continuations [avoid]. A new name may be a name that the
   expression binds. *)
Definition canonical_avoiding (avoid : list continuation) (e : expr)
    : expr :=
  snd
    (apart identity_renaming
       (fresh_names (binders e)
          (avoid ++ free_variables_of_expr e ++ free_continuations_of_expr e))
       e).

(* ======================================================================== *)
(* The names that a term binds                                              *)
(* ======================================================================== *)

(* The names that the binders of an expression bind, variables and
   continuations, in its code too, in the order in which
   FlambdaRenameApartSpec.apart gives them new names. There are
   [FlambdaRenameApartSpec.binders] of them. *)
Fixpoint bound_names (e : expr) : list positive :=
  match e with
  | Let x _ _ body => x :: bound_names body
  | Let_static bindings body =>
    flat_map bound_names_in_binding bindings ++ bound_names body
  | Let_cont_non_recursive k h body =>
    bound_names_in_handler h ++ k :: bound_names body
  | Let_cont_recursive handlers body =>
    map fst handlers
    ++ flat_map (fun kh => bound_names_in_handler (snd kh)) handlers
    ++ bound_names body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => []
  end
with bound_names_in_handler (h : continuation_handler) : list positive :=
  match h with
  | Handler params _ handler => params ++ bound_names handler
  end
with bound_names_in_binding (binding : static_binding) : list positive :=
  match binding with
  | Bind_code _ c => bound_names_in_code c
  | Bind_deleted_code _ | Bind_symbol _ _ => []
  end
with bound_names_in_code (c : code) : list positive :=
  match c with
  | Code params my_closure return_continuation exn_continuation _ body =>
    my_closure :: params
    ++ return_continuation :: exn_continuation :: bound_names body
  end.

(* The names that the expression binds are different from one another
   and from the names [outside]. A name that is bound twice, be it once
   as a variable and once as a continuation, is not different from
   itself. *)
Definition binders_apart (outside : list positive) (e : expr) : bool :=
  distinct (bound_names e)
  && forallb (fun name => negb (among name outside)) (bound_names e).

(* ======================================================================== *)
(* Statements                                                               *)
(* ======================================================================== *)

(* Alpha-equivalent expressions have the same canonical form: an
   equality of expressions, with no premise. *)
Definition alpha_equivalent_canonical_statement : Prop :=
  ∀ (avoid : list continuation) (e e' : expr),
    alpha_equivalent e e' ->
    canonical_avoiding avoid e = canonical_avoiding avoid e'.

(* The names that the candidates bind are the new names, in order. *)
Definition renamed_apart_avoiding_bound_names_statement : Prop :=
  ∀ (avoid : list continuation) (e : expr),
    bound_names (renamed_apart_avoiding avoid e)
    = fresh_names (binders e) (avoid ++ names_of e).

Definition canonical_avoiding_bound_names_statement : Prop :=
  ∀ (avoid : list continuation) (e : expr),
    bound_names (canonical_avoiding avoid e)
    = fresh_names (binders e)
        (avoid ++ free_variables_of_expr e ++ free_continuations_of_expr e).

(* So the binders of the candidates are apart. *)
Definition renamed_apart_avoiding_binders_apart_statement : Prop :=
  ∀ (avoid : list continuation) (e : expr),
    binders_apart (avoid ++ names_of e) (renamed_apart_avoiding avoid e)
    = true.

Definition canonical_avoiding_binders_apart_statement : Prop :=
  ∀ (avoid : list continuation) (e : expr),
    binders_apart
      (avoid ++ free_variables_of_expr e ++ free_continuations_of_expr e)
      (canonical_avoiding avoid e)
    = true.

(* On a term whose code is closed and whose binders are distinct, the
   canonical form is the term up to bound names. Both premises are
   needed, as for FlambdaRenameApartSpec.renamed_apart_avoiding
   (FlambdaRenameApartAlphaSpec.v). *)
Definition canonical_avoiding_alpha_statement : Prop :=
  ∀ (avoid : list continuation) (e : expr),
    code_closed e = true ->
    binders_distinct e = true ->
    alpha_equivalent e (canonical_avoiding avoid e).

(* So two such terms are alpha-equivalent exactly when their canonical
   forms are equal. *)
Definition alpha_equivalent_iff_canonical_statement : Prop :=
  ∀ (avoid : list continuation) (e e' : expr),
    code_closed e = true ->
    binders_distinct e = true ->
    code_closed e' = true ->
    binders_distinct e' = true ->
    (alpha_equivalent e e'
     <-> canonical_avoiding avoid e = canonical_avoiding avoid e').

(* The canonical form of such a term is its own canonical form. *)
Definition canonical_avoiding_idempotent_statement : Prop :=
  ∀ (avoid : list continuation) (e : expr),
    code_closed e = true ->
    binders_distinct e = true ->
    canonical_avoiding avoid (canonical_avoiding avoid e)
    = canonical_avoiding avoid e.

(* It behaves as the term does, and the comparison up to bound names
   accepts it: a check of it at each run would not fail. *)
Definition canonical_avoiding_equivalent_statement : Prop :=
  ∀ (avoid : list continuation) (e : expr),
    code_closed e = true ->
    binders_distinct e = true ->
    expr_equivalent e (canonical_avoiding avoid e).

Definition canonical_avoiding_accepted_statement : Prop :=
  ∀ (avoid : list continuation) (e : expr),
    code_closed e = true ->
    binders_distinct e = true ->
    expr_eqb [] [] e (canonical_avoiding avoid e) = true.

(* The free names of the candidates are among those of the term. No
   premise: a candidate that is not the term up to bound names has no
   other free name than the term all the same. *)
Definition renamed_apart_avoiding_free_names_statement : Prop :=
  ∀ (avoid : list continuation) (e : expr),
    incl (free_variables_of_expr (renamed_apart_avoiding avoid e))
      (free_variables_of_expr e)
    ∧ incl (free_continuations_of_expr (renamed_apart_avoiding avoid e))
        (free_continuations_of_expr e).

Definition canonical_avoiding_free_names_statement : Prop :=
  ∀ (avoid : list continuation) (e : expr),
    incl (free_variables_of_expr (canonical_avoiding avoid e))
      (free_variables_of_expr e)
    ∧ incl (free_continuations_of_expr (canonical_avoiding avoid e))
        (free_continuations_of_expr e).

(* The canonical form of a renamed term is the canonical form of the
   term under a second renaming, which does to the free names of that
   form, and to the continuations to avoid, what the first does. The
   second renaming sends each new name of the term to the new name at
   the same place of the renamed term. No premise. *)
Definition canonical_avoiding_equivariant_up_to_statement : Prop :=
  ∀ (r : renaming) (avoid : list continuation) (e : expr),
    ∃ r' : renaming,
      (∀ x : variable,
        among x (free_variables_of_expr (canonical_avoiding avoid e)) = true
        ->
        rename_variable r' x = rename_variable r x)
      ∧ (∀ k : continuation,
          among k (free_continuations_of_expr (canonical_avoiding avoid e))
          = true ->
          rename_continuation r' k = rename_continuation r k)
      ∧ (∀ k : continuation,
          among k avoid = true ->
          rename_continuation r' k = rename_continuation r k)
      ∧ canonical_avoiding (map (rename_continuation r) avoid)
          (rename_expr r e)
        = rename_expr r' (canonical_avoiding avoid e).

(* The form of FlambdaRenameApartSpec
   .rename_apart_equivariant_up_to_names_statement, in which the second
   renaming does to every name of the term, bound names too, what the
   first does, is false of the canonical form: a new name may be a
   name that the term binds. *)
Definition canonical_avoiding_not_equivariant_up_to_names_statement : Prop :=
  ~ (∀ (r : renaming) (e : expr),
      ∃ r' : renaming,
        (∀ x : variable,
          among x (variables_of e) = true ->
          rename_variable r' x = rename_variable r x)
        ∧ (∀ k : continuation,
            among k (continuations_of e) = true ->
            rename_continuation r' k = rename_continuation r k)
        ∧ canonical_avoiding [] (rename_expr r e)
          = rename_expr r' (canonical_avoiding [] e)).

(* The candidates of a term whose code is closed have their code
   closed. No premise on the binders. *)
Definition renamed_apart_avoiding_code_closed_statement : Prop :=
  ∀ (avoid : list continuation) (e : expr),
    code_closed e = true ->
    code_closed (renamed_apart_avoiding avoid e) = true.

Definition canonical_avoiding_code_closed_statement : Prop :=
  ∀ (avoid : list continuation) (e : expr),
    code_closed e = true ->
    code_closed (canonical_avoiding avoid e) = true.

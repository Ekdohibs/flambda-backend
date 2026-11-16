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

(* What the first candidate of the renaming apart makes is the term up
   to bound names.

   This file is to be reviewed: the two statements.

   FlambdaRenameApartSpec.rename_apart_avoiding makes a candidate, the
   term with every binder given a new name, and keeps it only if the
   comparison up to bound names (FlambdaCompareSpec.expr_eqb) says, at
   that run, that it is the term. Its statements hold whatever the
   candidate is, by the soundness of the comparison. The candidate was
   FlambdaRenameApartSpec.renamed_apart_avoiding, whose new names are
   different from every name of the term; it is now
   FlambdaRenameApartSpec.canonical_avoiding, whose new names are
   different from the free names of the term, and
   FlambdaRenameApartCanonicalSpec.v says the same of that one, from
   what is proved here. This file says what the comparison finds of
   the first candidate, once for all:

   - [renamed_apart_avoiding_alpha_statement]: the candidate of a term
     whose code is closed and whose binders are distinct is
     alpha-equivalent to the term (FlambdaAlphaSpec.alpha_equivalent).
     The two premises are rows of well-formedness
     (FlambdaBindersSpec.code_closed, [binders_distinct]): code
     mentions no name but those it binds, and the names that one
     binder binds are distinct.
   - That on such a term the normalisation gives its candidate, so
     that the check made at each run does not fail on a well-formed
     term, is said of the candidate that the normalisation has now
     (FlambdaRenameApartCanonicalSpec
     .rename_apart_avoiding_candidate_statement). It was a statement
     of this file when the candidate was the first.
   - [renamed_apart_repeated_parameter_statement]: distinct binders
     are needed. The candidate has a new name for each parameter of a
     handler; for a handler that repeats a parameter those are two
     names where the term has one, and alpha-equivalent terms have
     their binders distinct together
     (FlambdaAlphaSpec.alpha_equivalent_binders_distinct_statement).
     The comparison accepts that candidate all the same
     (FlambdaCompareAlphaSpec.expr_eqb_repeated_parameter_statement
     is the same pair of handlers).

   Closed code is asked because every rule of [alpha_equivalent] that
   renames a binder asks it of the scope: a renaming goes into the
   bodies of code, and code that mentions a name bound outside it has
   that name renamed with the binder, which no rule allows.

   Nothing is trusted here. The proofs are in
   FlambdaRenameApartAlphaProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaBindersSpec
  FlambdaCompareSpec FlambdaNominalSpec FlambdaAlphaSpec
  FlambdaRenameApartSpec.
Import ListNotations.

(* The candidate, whatever the continuations that its new names are to
   avoid. *)
Definition renamed_apart_avoiding_alpha_statement : Prop :=
  ∀ (avoid : list continuation) (e : expr),
    code_closed e = true ->
    binders_distinct e = true ->
    alpha_equivalent e (renamed_apart_avoiding avoid e).

(* A term whose code is closed, whose candidate is not
   alpha-equivalent to it, and whose candidate the comparison accepts:
   a handler repeats a parameter. *)
Definition renamed_apart_repeated_parameter_statement : Prop :=
  ∃ e : expr,
    code_closed e = true
    ∧ binders_distinct e = false
    ∧ ~ alpha_equivalent e (renamed_apart_avoiding [] e)
    ∧ expr_eqb [] [] e (renamed_apart_avoiding [] e) = true.

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

(* The validation asked again, with the aliases followed by a function
   that knows what symbols are bound to.

   This file is to be reviewed: the definitions and the statements.

   FlambdaValidateSpec.validate_unit normalises the term before
   Simplify and the term after in rounds, tidies each once its
   continuations are placed, and compares them. In each round, and
   where it tidies, it follows aliases with the function of
   FlambdaAliasesSpec.v, which knows nothing of what a
   symbol is bound to. The function of
   FlambdaEnvAliasesFollowSpec.v knows the closures that
   the static bindings in sight bind symbols to: it makes an indirect
   call of such a symbol the direct call of its code, puts in the
   place of the projection of a value slot from such a symbol what the
   slot holds, and in the place of the projection of a function slot
   the symbol that the set names for that slot. It also knows the
   closure that a [Let] binds a variable to when the closure is one of
   a set whose function slots are named, which the first does not.
   Units that are not validated with the first are validated with the
   second: those in which Simplify has used what a symbol is bound to,
   or what such a variable is bound to.

   The second function does not go into the bodies of code. Here it is
   run on the term and then on the body of each piece of code that the
   term binds ([follow_aliases_knowing]), in a body from nothing
   known: what the term around the code knows of a symbol is not used
   in the body, and what a [Let] of the body binds is.

   The second does not take the place of the first in the rounds or
   in the tidying. The normalisation of the term before has more
   rewrites than that of the term after: it may refine, and the other
   must keep the behaviour.
   Where the second function makes a call direct in both terms, the
   normalisation of the term before goes on from there (it inlines the
   call, moves the continuations that are applied once to where they
   are applied, and folds), and that of the term after may stop
   earlier (it inlines a call only where the kinds of the arguments
   are sure, and leaves a continuation that is applied once where it
   is). A unit whose two terms are the same once normalised by the
   first function, both left with the same indirect calls, is then
   not validated by the second. The example
   middle_end/flambda2/tests/semantics/indirect_higher_order.fl is
   one: with its two calls of a symbol made direct and inlined in
   both terms, the term before is folded to a constant, and the term
   after is left with a continuation applied once to a constant.

   So the validation is asked twice. First as it is. Then, if it
   fails, on the two terms normalised by rounds in front of each of
   which the second function is run ([normalise_knowing]): the round
   of the term before for the term before, the round of the term
   after for the term after. With a round that refines, the term so
   normalised refines the term, and with a round that is an
   equivalence it is equivalent to the term
   ([normalise_knowing_refines_statement],
   [normalise_knowing_equivalent_statement]); so if the second of the
   two normalised terms refines the first, by whatever
   FlambdaValidateSpec.v accepts, the term after refines the term
   before ([validate_knowing_sound_statement], for a unit run on its
   own from empty environments, and
   [validate_knowing_sound_everywhere_statement], under all
   environments). Nothing of FlambdaValidateSpec.v changes, and what it
   validates today it validates still.

   In those statements the rounds are arguments, and [validate_unit]
   is not named: they say what follows from what it concludes, for any
   round that refines and any round that is an equivalence.

   The second asking itself is [validate_unit_knowing]: [validate_unit]
   of the two terms so normalised, the term before with
   FlambdaValidateSpec.normalise_before_round_avoiding, which refines
   (FlambdaValidateProofs.normalise_before_round_avoiding_refines), and
   the term after with FlambdaValidateSpec.normalise_after_round, which
   is an equivalence (normalise_after_round_equivalent). The round of
   the term before makes names, and is told not to make one that is a
   continuation of the unit of a lifting
   (FlambdaValidateSpec.unit_continuations) or the continuation given
   for a loop. It is the means that [validate_unit] uses for its
   rounds after a lifting, which are told to avoid the continuations
   of the units of the liftings: a lifting is made only if what it
   makes is a well-formed unit, which goes by the names of the
   continuations of its unit. Those rounds are not told the
   continuations of the loops, since inside [validate_unit] the loops
   are made before any round. Here they are still to be made
   (FlambdaLoopifySpec.loopify_unit, which binds the continuation of
   each loop, runs inside [validate_unit], after this normalisation),
   so the round avoids those continuations too. What it accepts is
   correct of the two terms as they were given
   ([validate_unit_knowing_sound_statement],
   [validate_unit_knowing_sound_everywhere_statement]). So what the
   test tool reports as validated at the second asking is correct by
   one theorem, as what it reports at the first is, and not by how the
   tool puts proved pieces together: the tool calls
   [validate_unit_knowing] on the two terms as it converted them. The
   guesses that it passes (the static bindings to remove, the loops,
   the liftings, the value slots) are not trusted, here as there. The
   function takes the arguments of [validate_unit] and of the two
   rounds; whoever gives one of those an argument more gives it here
   too.

   Renaming and scope. The validation is stated of terms up to their
   bound names, so a function on terms that it runs is to commute with
   renaming, at least up to the names that it makes, and to keep scope
   (FlambdaValidatorEquivarianceSpec.v). [normalise_knowing] does, for
   a round that does: [normalise_knowing_keeps_scope_statement],
   [normalise_knowing_equivariant_statement] and
   [normalise_knowing_equivariant_up_to_statement]; and, for a round
   that is given continuations that the names it makes are to avoid,
   [normalise_knowing_equivariant_up_to_avoiding_statement]. At one
   round they are the same of [knowing_round]. They rest on the laws
   of the function that follows aliases
   (FlambdaEnvAliasesFollowSpec.rename_follow_aliases_empty_statement,
   FlambdaEnvAliasesFollowScopeSpec.v), on those of a rewrite taken
   into the bodies of code (FlambdaKeepsScopeSpec.v and
   FlambdaValidatorEquivarianceSpec.v, for [rewrite_code]) and on
   those of composition and iteration of
   FlambdaValidatorEquivarianceSpec.v. They are laws of
   the normalisation. What follows from them for the second asking is
   that its verdict does not depend on the names of the unit, where
   the code of the term before is closed
   ([validate_unit_knowing_equivariant_statement]); the round of the
   term before avoids the continuations named above for that.

   The proofs are in FlambdaValidateKnowingProofs.v, which needs no
   review. *)

(* CR mshinwell: the function of FlambdaAliasesSpec.v stays in the
   rounds and in the tidying of the two terms
   (FlambdaValidateSpec.tidied_before, tidied_after), and the
   validation is asked a second time with the new one in front of
   each round. The first can go, and the new one take its place in
   both, when two things are done. The normalisation of
   the term after is to reach what that of the term before reaches on
   indirect_higher_order.fl once the calls are direct, or the term
   before is to keep the calls that Simplify kept (the list of code
   that is not to be inlined), so that the example is validated with
   the new function alone. And the pass for loops
   (FlambdaLoopAliasesSpec.v), which is written with the typing
   environment of FlambdaAliasesSpec.v, is to be written with what the
   new function knows. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaSemanticsSpec
  FlambdaNominalSpec FlambdaBindersSpec FlambdaNormaliseSpec
  FlambdaUnusedStaticsSpec FlambdaLiftSpec FlambdaLoopifySpec
  FlambdaValidateSpec FlambdaValidatorEquivarianceSpec.
From OxCamlFlambda2Types Require FlambdaEnvAliasesFollowSpec.
Import ListNotations.

(* The aliases followed with what is known of the symbols and of the
   closures that a [Let] binds: in the term, and then in the body of
   each piece of code that the term binds, where it starts from nothing
   known (FlambdaNormaliseSpec.rewrite_code, which leaves a body alone
   if the rewrite would change whether it binds something static). *)
Definition follow_aliases_knowing (e : expr) : expr :=
  rewrite_code
    (FlambdaEnvAliasesFollowSpec.follow_aliases
       FlambdaEnvAliasesFollowSpec.Known.empty)
    (FlambdaEnvAliasesFollowSpec.follow_aliases
       FlambdaEnvAliasesFollowSpec.Known.empty e).

(* A round, after the aliases have been followed so. *)
Definition knowing_round (round : expr -> expr) (e : expr) : expr :=
  round (follow_aliases_knowing e).

Definition normalise_knowing (round : expr -> expr) (rounds : nat)
    (e : expr) : expr :=
  iterate rounds (knowing_round round) e.

(* With a round that refines, it refines the term; with a round that is
   an equivalence, it is equivalent to the term. *)
Definition normalise_knowing_refines_statement : Prop :=
  ∀ (round : expr -> expr) (rounds : nat) (e : expr),
    (∀ e : expr, expr_refines e (round e)) ->
    expr_refines e (normalise_knowing round rounds e).

Definition normalise_knowing_equivalent_statement : Prop :=
  ∀ (round : expr -> expr) (rounds : nat) (e : expr),
    (∀ e : expr, expr_equivalent e (round e)) ->
    expr_equivalent e (normalise_knowing round rounds e).

(* If the term after, so normalised, refines the term before, so
   normalised, then the term after refines the term before: run on its
   own from empty environments, and under all environments. The premise
   is what FlambdaValidateSpec.validate_unit concludes of the two
   normalised terms where it validates them
   (FlambdaValidateSpec.validate_unit_sound_statement,
   validate_unit_sound_everywhere_statement). *)
Definition validate_knowing_sound_statement : Prop :=
  ∀ (round_before round_after : expr -> expr) (rounds : nat)
    (before after : expr),
    (∀ e : expr, expr_refines e (round_before e)) ->
    (∀ e : expr, expr_equivalent e (round_after e)) ->
    program_refines (normalise_knowing round_before rounds before)
      (normalise_knowing round_after rounds after) ->
    program_refines before after.

Definition validate_knowing_sound_everywhere_statement : Prop :=
  ∀ (round_before round_after : expr -> expr) (rounds : nat)
    (before after : expr),
    (∀ e : expr, expr_refines e (round_before e)) ->
    (∀ e : expr, expr_equivalent e (round_after e)) ->
    expr_refines (normalise_knowing round_before rounds before)
      (normalise_knowing round_after rounds after) ->
    expr_refines before after.

(* ======================================================================== *)
(* The second asking                                                        *)
(* ======================================================================== *)

(* The validation asked the second time: [validate_unit], with the same
   arguments, of the two terms normalised as above, the term before
   with the round of the term before and the term after with the round
   of the term after. The round of the term before avoids, for the
   names that it makes, the continuations of the units of the liftings
   and those given for the loops. *)
Definition validate_unit_knowing (rounds : nat) (unused : dropped)
    (slots : list value_slot) (G : loopified) (Ls : list lifting)
    (kept : list code_id) (before after : expr) : validation :=
  validate_unit rounds unused slots G Ls kept
    (normalise_knowing
       (normalise_before_round_avoiding
          (unit_continuations Ls ++ map snd G) kept)
       rounds before)
    (normalise_knowing (normalise_after_round kept) rounds after).

(* A unit that it validates refines the unit it came from, run on its
   own from empty environments. *)
Definition validate_unit_knowing_sound_statement : Prop :=
  ∀ (rounds : nat) (unused : dropped) (slots : list value_slot)
    (G : loopified) (Ls : list lifting) (kept : list code_id)
    (before after : expr),
    validate_unit_knowing rounds unused slots G Ls kept before after
    <> Not_validated ->
    program_refines before after.

(* It does so under all environments if no static binding was removed
   and nothing was lifted. *)
Definition validate_unit_knowing_sound_everywhere_statement : Prop :=
  ∀ (rounds : nat) (unused : dropped) (slots : list value_slot)
    (G : loopified) (Ls : list lifting) (kept : list code_id)
    (before after : expr),
    validate_unit_knowing rounds unused slots G Ls kept before after
    = Same_up_to_bound_names
    ∨ validate_unit_knowing rounds unused slots G Ls kept before after
      = Same_once_normalised ->
    expr_refines before after.

(* It answers the same on a pair of terms and on the pair renamed, the
   names given for the loops and the liftings being renamed with them,
   where the code of the term before is closed: what
   FlambdaValidatorEquivarianceSpec.validate_unit_equivariant_statement
   says of [validate_unit]. The round of the term before makes names,
   so the term that [validate_unit] is given here is, for a renamed
   pair, the renamed one only up to those names; and the liftings and
   the loops speak of continuations that the term need not mention.
   The round is told to avoid those continuations, so the renaming
   that [normalise_knowing_equivariant_up_to_avoiding_statement] gives
   does to them what the first does, which is what
   FlambdaValidatorEquivarianceSpec.validate_unit_renamed_alike_statement
   asks. The round of the term after makes no name. Nothing is said of
   a term before with open code. The two statements above do not rest
   on this one: a unit validated at the second asking is correct. *)
Definition validate_unit_knowing_equivariant_statement : Prop :=
  ∀ (r : renaming) (rounds : nat) (unused : dropped)
    (slots : list value_slot) (G : loopified) (Ls : list lifting)
    (kept : list code_id) (before after : expr),
    code_closed before = true ->
    validate_unit_knowing rounds unused slots (rename_loopified r G)
      (map (rename_lifting r) Ls) kept (rename_expr r before)
      (rename_expr r after)
    = validate_unit_knowing rounds unused slots G Ls kept before after.

(* ======================================================================== *)
(* Renaming and scope                                                       *)
(* ======================================================================== *)

(* With a round that keeps scope, it keeps scope. *)
Definition normalise_knowing_keeps_scope_statement : Prop :=
  ∀ (round : expr -> expr) (rounds : nat),
    keeps_scope round -> keeps_scope (normalise_knowing round rounds).

(* With a round that commutes with renaming, it commutes with
   renaming. *)
Definition normalise_knowing_equivariant_statement : Prop :=
  ∀ (round : expr -> expr) (rounds : nat),
    rewrite_equivariant round ->
    rewrite_equivariant (normalise_knowing round rounds).

(* With a round that keeps scope and commutes with renaming up to the
   names that it makes, it commutes with renaming up to the names that
   it makes. *)
Definition normalise_knowing_equivariant_up_to_statement : Prop :=
  ∀ (round : expr -> expr) (rounds : nat),
    keeps_scope round ->
    equivariant_up_to round ->
    equivariant_up_to (normalise_knowing round rounds).

(* The same of a round that is given continuations that the names it
   makes are to avoid
   (FlambdaValidatorEquivarianceSpec.equivariant_up_to_avoiding): with
   a round that keeps scope whatever it is given and commutes with
   renaming up to the names that it makes, the rounds do, given the
   same continuations at every round, and the second renaming does to
   those continuations what the first does. *)
Definition normalise_knowing_equivariant_up_to_avoiding_statement : Prop :=
  ∀ (round : list continuation -> expr -> expr) (rounds : nat),
    (∀ avoid : list continuation, keeps_scope (round avoid)) ->
    equivariant_up_to_avoiding round ->
    equivariant_up_to_avoiding
      (fun avoid => normalise_knowing (round avoid) rounds).

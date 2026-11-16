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

(* The validation asked of the term before with copies of code for the
   sets of closures that a [Let] makes.

   This file is to be reviewed: [lifted_steps], [copied_avoiding],
   [before_copied], [validate_unit_copying],
   [validate_unit_knowing_copying], [lifted_steps_free_in] and
   [before_copied_holds_from], which are extracted and run by the
   validation of middle_end/flambda2/tests/semantics/; and the
   statements. No definition of another file is changed for it, and
   no function of another file of the model calls one of this file:
   the test tool does.

   Why it is a function beside, and of what term.
   FlambdaCopyLetsSpec.with_let_copies gives the [Let]s at the paths
   it is told a copy of their code. The [Let] that Simplify gave new
   code is, most often, one that the term before as it is given has
   not: Simplify gave the set new code where it inlined the function
   that makes the set, and the term before has that [Let] once, in
   the code of the function. The rounds that follow a lifting inline
   the call and bring the [Let] out
   (FlambdaValidateSpec.normalise_lifted_own). So the copies are made
   of the term before as some liftings and their rounds have made it:

   - [lifted_steps] is the liftings given, in turn, each followed by
     its rounds: FlambdaValidateSpec.lifted_in_turn without what that
     function does after the last lifting;
   - [before_copied rounds G Ls L copies Ls' kept before] is the term
     before with its loops made, as FlambdaValidateSpec.validate_unit
     makes them first; then [lifted_steps] of the liftings [Ls]; then
     the lifting [L], with no round after it
     (FlambdaValidateSpec.with_lifting); then the copies. The paths of
     the copies are paths in the term that [L] gives;
   - [validate_unit_copying] is FlambdaValidateSpec.validate_unit, of
     that term and of the term after, with the liftings [Ls'], which
     are liftings of that term; [validate_unit_knowing_copying] is
     FlambdaValidateKnowingSpec.validate_unit_knowing of the same.
     The function asked begins again from the term that it is given:
     it makes the loops, normalises, compares, and makes its own
     liftings.
   The rounds of [lifted_steps] are told not to make a name that is
   one of the continuations of the units of the liftings, of [Ls],
   of [L] and of [Ls'], or of the loops ([copied_avoiding]): the
   function asked reads those after, and the term need not mention
   them (FlambdaValidateReadingSpec.before_read is given its
   continuations for the same reason).

   What they accept is correct for the unit run on its own, and not
   under all environments. Each step of [before_copied] refines the
   unit run on its own from empty environments: the loops, each
   lifting where its tests pass, the rounds, and the copies where
   their tests pass (FlambdaCopyLetsSpec.v). So [before_copied]
   refines the term before so run
   ([before_copied_program_refines_statement]), and a term after that
   the function asked accepts for the term that [before_copied] gives
   refines the term before so run
   ([validate_unit_copying_sound_statement],
   [validate_unit_knowing_copying_sound_statement]). A lifting and a
   copy are not refinements under all environments, so nothing here
   is stated under all environments, whatever the verdict.

   From the imports. A lifting refines the unit run from its imports
   where the static environment of the imports binds no name that the
   unit lifted binds, and the rounds have tests of the same kind
   (FlambdaValidateImportsSpec.lifting_free_in,
   normalise_lifted_own_free_in). [before_copied_holds_from] is those
   tests of the liftings [Ls] and [L], on the terms that
   [before_copied] applies them to; the copies ask nothing of the
   imports. Where it passes, [before_copied] refines the term before
   run from its imports ([before_copied_refines_imports_statement]);
   and where the test of FlambdaValidateImportsSpec.v passes too, of
   the verdict that the function asked gives of the term that
   [before_copied] gives, the term after refines the term before, each
   run from the imports
   ([validate_unit_copying_verdict_imports_statement],
   [validate_unit_knowing_copying_verdict_imports_statement]).

   What the test tool prints, and by which statement. A caller asks
   the functions of FlambdaValidateSpec.v and its like first, and
   these only of a unit that they do not validate; the lists of
   liftings and the copies are its guess, and nothing is believed of
   them. For a unit that one of the two functions here accepts, the
   tool prints the verdict line of the function asked, and under it
   the line

     (asked of the term before with copies of code for the sets of
     closures that a Let makes)

   on one line. The verdict line begins "validated:" for a unit that
   imports nothing, by [validate_unit_copying_sound_statement] or
   [validate_unit_knowing_copying_sound_statement]. For a unit that
   has imports it begins "validated with its imports:" where
   [before_copied_holds_from] and the test of the verdict both pass,
   by [validate_unit_copying_verdict_imports_statement] or
   [validate_unit_knowing_copying_verdict_imports_statement]; and
   "validated with nothing imported:" otherwise, which is then of the
   two terms run from empty environments, by the two statements
   first named. Where the two terms have in front of them a prefix
   made from the .cmx files of other units, the verdict line says so
   in the words that the tool has for every verdict, and the
   statements are of the two terms with the prefix. That the tool
   calls these functions on the terms that it reads, and prints so,
   is not proved (rocq/TRUST-LEDGER.md).

   Under renaming. The two functions answer the same of a pair of
   terms and of the pair renamed, the loops and the three lots of
   liftings being renamed with them, where the code of the term
   before is closed ([validate_unit_copying_equivariant_statement],
   [validate_unit_knowing_copying_equivariant_statement]); on the way,
   [lifted_steps] and [before_copied] keep scope and commute with
   renaming up to the names that the rounds make, in the forms of
   FlambdaValidatorEquivarianceSpec.v, and the copies keep scope. The
   copies are not renamed: they name code ids, symbols and places.
   The tests from the imports answer the same of the renamed terms
   too: [before_copied_holds_from]
   ([before_copied_holds_from_equivariant_statement]) and, together
   with it, the test of the verdict that the function asked gives
   ([validate_unit_copying_holds_from_equivariant_statement],
   [validate_unit_knowing_copying_holds_from_equivariant_statement],
   in the form of FlambdaValidateImportsRenameSpec
   .validate_unit_holds_from_equivariant_statement). So whether a
   unit that one of the two functions accepts is validated with its
   imports does not depend on the names of its variables and of its
   continuations.

   What is not said. Nothing is said of the two terms under all
   environments, or followed by what may follow a unit: the second
   check of a unit (FlambdaValidateRelatedSpec.v) does not ask for the
   copies. A lifting lifts nothing from a term that has a primitive
   of the heap (the guard at FlambdaLiftSpec.lift_unit and its like),
   and no copy for a [Let] is made of one. Each tests the term that
   it is given, which is the term before as the steps before it have
   made it, and not the term before as it came. Of such a term the
   two functions say what the function asked says of it. Nothing is
   said of the regions of a unit accepted here
   (FlambdaRegionsCheckSoundSpec.v states its second form of
   FlambdaValidateSpec.validate_unit alone). Of a term before with
   open code only what is stated without that premise is said. The
   statements about the functions under renaming are about the
   functions as they are defined, and not about what is proved of a
   pair that they accept.

   The proofs are in FlambdaValidateCopyingProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaNominalSpec FlambdaBindersSpec
  FlambdaWellFormedSpec
  FlambdaUnusedStaticsSpec FlambdaLoopifySpec FlambdaStubCallsSpec
  FlambdaNeverReturnsSpec FlambdaLiftSpec FlambdaCopyLetsSpec
  FlambdaValidateSpec FlambdaValidatorEquivarianceSpec
  FlambdaValidateKnowingSpec FlambdaImportsSpec
  FlambdaValidateImportsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The term that is given the copies                                        *)
(* ======================================================================== *)

(* Each of the liftings given, in turn, each followed by its rounds:
   FlambdaValidateSpec.lifted_in_turn, without what that function does
   after the last. With no lifting it is the term. *)
Fixpoint lifted_steps (rounds : nat) (avoid : list continuation)
    (Ls : list lifting) (kept : list code_id) (e : expr) : expr :=
  match Ls with
  | [] => e
  | L :: Ls =>
    lifted_steps rounds avoid Ls kept
      (normalise_lifted_own (made_closures_called L) (own_closures_known L)
         rounds avoid kept (with_lifting L e))
  end.

(* The continuations that the rounds of [lifted_steps] are told not to
   name: those of the units of all the liftings, and those of the
   loops. *)
Definition copied_avoiding (G : loopified) (Ls : list lifting) (L : lifting)
    (Ls' : list lifting) : list continuation :=
  unit_continuations (Ls ++ L :: Ls') ++ map snd G.

(* The term before with its loops made, as
   FlambdaValidateSpec.validate_unit makes them; the liftings [Ls],
   each with its rounds; the lifting [L], with no round; and then the
   copies. [Ls'] is given for its continuations only. *)
Definition before_copied (rounds : nat) (G : loopified) (Ls : list lifting)
    (L : lifting) (copies : list let_copy) (Ls' : list lifting)
    (kept : list code_id) (before : expr) : expr :=
  with_let_copies copies
    (with_lifting L
       (lifted_steps rounds (copied_avoiding G Ls L Ls') Ls kept
          (loopify_unit G (cut_returns [] (demote_stub_calls before))))).

(* ======================================================================== *)
(* The two functions asked                                                  *)
(* ======================================================================== *)

(* FlambdaValidateSpec.validate_unit of that term and the term after,
   with the liftings [Ls']. *)
Definition validate_unit_copying (rounds : nat) (unused : dropped)
    (slots : list value_slot) (G : loopified) (Ls : list lifting)
    (L : lifting) (copies : list let_copy) (Ls' : list lifting)
    (kept : list code_id) (before after : expr) : validation :=
  validate_unit rounds unused slots G Ls' kept
    (before_copied rounds G Ls L copies Ls' kept before) after.

(* The same of the second asking. *)
Definition validate_unit_knowing_copying (rounds : nat) (unused : dropped)
    (slots : list value_slot) (G : loopified) (Ls : list lifting)
    (L : lifting) (copies : list let_copy) (Ls' : list lifting)
    (kept : list code_id) (before after : expr) : validation :=
  validate_unit_knowing rounds unused slots G Ls' kept
    (before_copied rounds G Ls L copies Ls' kept before) after.

(* ======================================================================== *)
(* The tests from the imports                                               *)
(* ======================================================================== *)

(* The tests of FlambdaValidateImportsSpec.lifted_free_in, of the
   liftings as [lifted_steps] applies them: for each, the test of the
   lifting, of the term that it is applied to, and the test of the
   rounds that follow it, of the term that it gives. *)
Fixpoint lifted_steps_free_in (statics : static_env) (rounds : nat)
    (avoid : list continuation) (Ls : list lifting) (kept : list code_id)
    (e : expr) : bool :=
  match Ls with
  | [] => true
  | L :: Ls =>
    lifting_free_in statics L e
    && normalise_lifted_own_free_in statics (made_closures_called L)
         (own_closures_known L) rounds avoid kept (with_lifting L e)
    && lifted_steps_free_in statics rounds avoid Ls kept
         (normalise_lifted_own (made_closures_called L)
            (own_closures_known L) rounds avoid kept (with_lifting L e))
  end.

(* Those of [Ls], and the test of the lifting [L], of the term that
   [before_copied] applies it to. Nothing is asked for the loops, as
   FlambdaValidateImportsSpec.lifted_holds_from asks nothing for them,
   nor for the copies. *)
Definition before_copied_holds_from (statics : static_env) (rounds : nat)
    (G : loopified) (Ls : list lifting) (L : lifting) (Ls' : list lifting)
    (kept : list code_id) (before : expr) : bool :=
  let avoid := copied_avoiding G Ls L Ls' in
  let looped := loopify_unit G (cut_returns [] (demote_stub_calls before)) in
  lifted_steps_free_in statics rounds avoid Ls kept looped
  && lifting_free_in statics L (lifted_steps rounds avoid Ls kept looped).

(* ======================================================================== *)
(* The statements: what is accepted                                         *)
(* ======================================================================== *)

(* The liftings in turn, each with its rounds, refine the unit run on
   its own from empty environments, whatever the liftings and the
   continuations to avoid: a lifting whose tests fail is not made. *)
Definition lifted_steps_program_refines_statement : Prop :=
  ∀ (rounds : nat) (avoid : list continuation) (Ls : list lifting)
    (kept : list code_id) (e : expr),
    program_refines e (lifted_steps rounds avoid Ls kept e).

(* So does the term that the two functions give to the function that
   they ask. *)
Definition before_copied_program_refines_statement : Prop :=
  ∀ (rounds : nat) (G : loopified) (Ls : list lifting) (L : lifting)
    (copies : list let_copy) (Ls' : list lifting) (kept : list code_id)
    (before : expr),
    program_refines before
      (before_copied rounds G Ls L copies Ls' kept before).

(* A unit that is validated here refines the unit it came from, run on
   its own from empty environments. *)
Definition validate_unit_copying_sound_statement : Prop :=
  ∀ (rounds : nat) (unused : dropped) (slots : list value_slot)
    (G : loopified) (Ls : list lifting) (L : lifting)
    (copies : list let_copy) (Ls' : list lifting) (kept : list code_id)
    (before after : expr),
    validate_unit_copying rounds unused slots G Ls L copies Ls' kept before
      after
    <> Not_validated ->
    program_refines before after.

Definition validate_unit_knowing_copying_sound_statement : Prop :=
  ∀ (rounds : nat) (unused : dropped) (slots : list value_slot)
    (G : loopified) (Ls : list lifting) (L : lifting)
    (copies : list let_copy) (Ls' : list lifting) (kept : list code_id)
    (before after : expr),
    validate_unit_knowing_copying rounds unused slots G Ls L copies Ls' kept
      before after
    <> Not_validated ->
    program_refines before after.

(* Where the tests of the liftings pass at the static environment of
   the imports, the term that the two functions give refines the term
   before, each run on its own from those imports. *)
Definition before_copied_refines_imports_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (rounds : nat) (G : loopified)
    (Ls : list lifting) (L : lifting) (copies : list let_copy)
    (Ls' : list lifting) (kept : list code_id) (before : expr),
    before_copied_holds_from (imports_statics imports) rounds G Ls L Ls' kept
      before
    = true ->
    unit_refines imports before
      (before_copied rounds G Ls L copies Ls' kept before).

(* A unit whose verdict here the test of FlambdaValidateImportsSpec.v
   accepts at the static environment of its imports, of the term that
   the function asked was given for the term before ([before_copied])
   and of the term after, and for which the tests of the liftings that
   make that term pass there too, refines the unit it came from, each
   run on its own from those imports. *)
Definition validate_unit_copying_verdict_imports_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (rounds : nat)
    (unused : dropped) (slots : list value_slot) (G : loopified)
    (Ls : list lifting) (L : lifting) (copies : list let_copy)
    (Ls' : list lifting) (kept : list code_id) (before after : expr),
    before_copied_holds_from (imports_statics imports) rounds G Ls L Ls' kept
      before
    = true ->
    verdict_holds_from (imports_statics imports) rounds unused slots G Ls'
      kept (before_copied rounds G Ls L copies Ls' kept before) after
      (validate_unit_copying rounds unused slots G Ls L copies Ls' kept
         before after)
    = true ->
    unit_refines imports before after.

(* The same of the second asking. *)
Definition validate_unit_knowing_copying_verdict_imports_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (rounds : nat)
    (unused : dropped) (slots : list value_slot) (G : loopified)
    (Ls : list lifting) (L : lifting) (copies : list let_copy)
    (Ls' : list lifting) (kept : list code_id) (before after : expr),
    before_copied_holds_from (imports_statics imports) rounds G Ls L Ls' kept
      before
    = true ->
    verdict_knowing_holds_from (imports_statics imports) rounds unused slots
      G Ls' kept (before_copied rounds G Ls L copies Ls' kept before) after
      (validate_unit_knowing_copying rounds unused slots G Ls L copies Ls'
         kept before after)
    = true ->
    unit_refines imports before after.

(* ======================================================================== *)
(* The statements: under renaming                                           *)
(* ======================================================================== *)

(* The copies make no name and bind code only: they keep scope. *)
Definition with_let_copies_keeps_scope_statement : Prop :=
  ∀ copies : list let_copy, keeps_scope (with_let_copies copies).

(* The liftings in turn, each with its rounds, keep scope. *)
Definition lifted_steps_keeps_scope_statement : Prop :=
  ∀ (rounds : nat) (avoid : list continuation) (kept : list code_id)
    (Ls : list lifting),
    keeps_scope (lifted_steps rounds avoid Ls kept).

(* They commute with renaming up to the names that the rounds make,
   where the continuations to avoid have among them the two of the
   unit of each lifting: FlambdaValidatorEquivarianceSpec
   .lifted_in_turn_equivariant_up_to_statement, word for word, of
   [lifted_steps]. The liftings and the continuations to avoid are
   renamed by the first renaming; the term may be under another, which
   does to its free names and to the continuations to avoid what the
   first does. *)
Definition lifted_steps_equivariant_up_to_statement : Prop :=
  ∀ (r : renaming) (rounds : nat) (avoid : list continuation)
    (kept : list code_id) (Ls : list lifting),
    (∀ L : lifting,
      In L Ls ->
      among (unit_return (lifting_unit L)) avoid = true
      ∧ among (unit_exn (lifting_unit L)) avoid = true) ->
    ∀ (r1 : renaming) (e : expr),
      code_closed e = true ->
      renames_alike r r1 e ->
      rename_conts_alike r r1 avoid ->
      ∃ r' : renaming,
        renames_alike r r' (lifted_steps rounds avoid Ls kept e)
        ∧ rename_conts_alike r r' avoid
        ∧ lifted_steps rounds (map (rename_continuation r) avoid)
            (map (rename_lifting r) Ls) kept (rename_expr r1 e)
          = rename_expr r' (lifted_steps rounds avoid Ls kept e).

Definition before_copied_keeps_scope_statement : Prop :=
  ∀ (rounds : nat) (G : loopified) (Ls : list lifting) (L : lifting)
    (copies : list let_copy) (Ls' : list lifting) (kept : list code_id),
    keeps_scope (before_copied rounds G Ls L copies Ls' kept).

(* For a renamed term, with the loops and the liftings renamed, it is
   the term that the two functions give under a second renaming, which
   does to the free names of that term, and to the continuations of
   the units of all the liftings and of the loops, what the first
   does. *)
Definition before_copied_renamed_statement : Prop :=
  ∀ (r : renaming) (rounds : nat) (G : loopified) (Ls : list lifting)
    (L : lifting) (copies : list let_copy) (Ls' : list lifting)
    (kept : list code_id) (before : expr),
    code_closed before = true ->
    ∃ r' : renaming,
      renames_alike r r'
        (before_copied rounds G Ls L copies Ls' kept before)
      ∧ rename_conts_alike r r' (copied_avoiding G Ls L Ls')
      ∧ before_copied rounds (rename_loopified r G)
          (map (rename_lifting r) Ls) (rename_lifting r L) copies
          (map (rename_lifting r) Ls') kept (rename_expr r before)
        = rename_expr r'
            (before_copied rounds G Ls L copies Ls' kept before).

(* Each of the two functions answers the same on a pair of terms and
   on the pair renamed, the names given for the loops and the liftings
   being renamed with them, where the code of the term before is
   closed. *)
Definition validate_unit_copying_equivariant_statement : Prop :=
  ∀ (r : renaming) (rounds : nat) (unused : dropped)
    (slots : list value_slot) (G : loopified) (Ls : list lifting)
    (L : lifting) (copies : list let_copy) (Ls' : list lifting)
    (kept : list code_id) (before after : expr),
    code_closed before = true ->
    validate_unit_copying rounds unused slots (rename_loopified r G)
      (map (rename_lifting r) Ls) (rename_lifting r L) copies
      (map (rename_lifting r) Ls') kept (rename_expr r before)
      (rename_expr r after)
    = validate_unit_copying rounds unused slots G Ls L copies Ls' kept
        before after.

Definition validate_unit_knowing_copying_equivariant_statement : Prop :=
  ∀ (r : renaming) (rounds : nat) (unused : dropped)
    (slots : list value_slot) (G : loopified) (Ls : list lifting)
    (L : lifting) (copies : list let_copy) (Ls' : list lifting)
    (kept : list code_id) (before after : expr),
    code_closed before = true ->
    validate_unit_knowing_copying rounds unused slots (rename_loopified r G)
      (map (rename_lifting r) Ls) (rename_lifting r L) copies
      (map (rename_lifting r) Ls') kept (rename_expr r before)
      (rename_expr r after)
    = validate_unit_knowing_copying rounds unused slots G Ls L copies Ls'
        kept before after.

(* ======================================================================== *)
(* The statements: the tests from the imports, under renaming               *)
(* ======================================================================== *)

(* The tests of the liftings that make the term which is given the
   copies answer of a renamed term what they answer of the term, the
   loops and the three lots of liftings being renamed with it, where
   its code is closed. The static environment is the same on the two
   sides: a renaming moves no symbol and no code id. *)
Definition before_copied_holds_from_equivariant_statement : Prop :=
  ∀ (r : renaming) (statics : static_env) (rounds : nat) (G : loopified)
    (Ls : list lifting) (L : lifting) (Ls' : list lifting)
    (kept : list code_id) (before : expr),
    code_closed before = true ->
    before_copied_holds_from statics rounds (rename_loopified r G)
      (map (rename_lifting r) Ls) (rename_lifting r L)
      (map (rename_lifting r) Ls') kept (rename_expr r before)
    = before_copied_holds_from statics rounds G Ls L Ls' kept before.

(* Whether the verdict that [validate_unit_copying] gives of two terms
   holds of the unit run from a static environment: the two tests
   that [validate_unit_copying_verdict_imports_statement] asks, both
   together, answer the same of the two terms renamed, where the code
   of the term before is closed. It is what decides whether the tool
   prints "with its imports" or "with nothing imported". *)
Definition validate_unit_copying_holds_from_equivariant_statement : Prop :=
  ∀ (r : renaming) (statics : static_env) (rounds : nat) (unused : dropped)
    (slots : list value_slot) (G : loopified) (Ls : list lifting)
    (L : lifting) (copies : list let_copy) (Ls' : list lifting)
    (kept : list code_id) (before after : expr),
    code_closed before = true ->
    before_copied_holds_from statics rounds (rename_loopified r G)
      (map (rename_lifting r) Ls) (rename_lifting r L)
      (map (rename_lifting r) Ls') kept (rename_expr r before)
    && verdict_holds_from statics rounds unused slots (rename_loopified r G)
         (map (rename_lifting r) Ls') kept
         (before_copied rounds (rename_loopified r G)
            (map (rename_lifting r) Ls) (rename_lifting r L) copies
            (map (rename_lifting r) Ls') kept (rename_expr r before))
         (rename_expr r after)
         (validate_unit_copying rounds unused slots (rename_loopified r G)
            (map (rename_lifting r) Ls) (rename_lifting r L) copies
            (map (rename_lifting r) Ls') kept (rename_expr r before)
            (rename_expr r after))
    = before_copied_holds_from statics rounds G Ls L Ls' kept before
      && verdict_holds_from statics rounds unused slots G Ls' kept
           (before_copied rounds G Ls L copies Ls' kept before) after
           (validate_unit_copying rounds unused slots G Ls L copies Ls' kept
              before after).

(* The same of the second asking. *)
Definition validate_unit_knowing_copying_holds_from_equivariant_statement : Prop
  :=
  ∀ (r : renaming) (statics : static_env) (rounds : nat) (unused : dropped)
    (slots : list value_slot) (G : loopified) (Ls : list lifting)
    (L : lifting) (copies : list let_copy) (Ls' : list lifting)
    (kept : list code_id) (before after : expr),
    code_closed before = true ->
    before_copied_holds_from statics rounds (rename_loopified r G)
      (map (rename_lifting r) Ls) (rename_lifting r L)
      (map (rename_lifting r) Ls') kept (rename_expr r before)
    && verdict_knowing_holds_from statics rounds unused slots
         (rename_loopified r G) (map (rename_lifting r) Ls') kept
         (before_copied rounds (rename_loopified r G)
            (map (rename_lifting r) Ls) (rename_lifting r L) copies
            (map (rename_lifting r) Ls') kept (rename_expr r before))
         (rename_expr r after)
         (validate_unit_knowing_copying rounds unused slots
            (rename_loopified r G) (map (rename_lifting r) Ls)
            (rename_lifting r L) copies (map (rename_lifting r) Ls') kept
            (rename_expr r before) (rename_expr r after))
    = before_copied_holds_from statics rounds G Ls L Ls' kept before
      && verdict_knowing_holds_from statics rounds unused slots G Ls' kept
           (before_copied rounds G Ls L copies Ls' kept before) after
           (validate_unit_knowing_copying rounds unused slots G Ls L copies
              Ls' kept before after).

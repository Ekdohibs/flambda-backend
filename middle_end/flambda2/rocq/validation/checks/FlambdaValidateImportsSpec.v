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

(* The validation, of a unit that imports symbols.

   This file is to be reviewed: [verdict_holds_from] and
   [verdict_knowing_holds_from], which are extracted and read by the
   test tool of middle_end/flambda2/tests/semantics/, with what they
   are made of ([holds_from], [lifted_holds_from] and the tests that
   it puts together); [holds_from_imports], the three verdicts that
   are accepted under no condition; the ten statements of the
   validation; and [static_names_free_in_head_statement].

   FlambdaValidateSpec.validate_unit_sound_statement is of the two
   units run from the empty static environment
   (FlambdaUnusedStaticsSpec.program_refines). A unit that mentions a
   symbol of another unit is run from the static environment of its
   imports (FlambdaImportsSpec.imports_statics), and of that run the
   statement says nothing: in the empty environment the symbol is bound
   to nothing.

   [validate_unit_sound_imports_statement] is the same of the unit run
   from its imports (FlambdaImportsSpec.unit_refines), whatever they
   are, for three of the five ways in which a unit is validated: the
   two terms are the same up to bound names; they are once normalised;
   they are once the unused static bindings of the term before are
   removed as well ([holds_from_imports]). Why these three:
   - the normalisations of the two terms, the making of loops, the
     placing of continuations, the tidying of the terms so placed and
     the comparison are refinements or equivalences under all
     environments, so under that of the imports;
   - the removal of unused static bindings is a refinement from any
     environments in which the names removed occur nowhere
     (FlambdaUnusedStaticsSpec.drop_statics_refines_statement), and
     they occur nowhere in the static environment of the imports, which
     has no code and binds each symbol to something that holds no
     value.
   Nothing else is asked of the list of imports. A list may bind a
   symbol which the unit before binds too. A run of the unit before
   that reaches that binding is then undefined (a static binding of a
   name that is bound has no step), and the statement holds of such a
   run for nothing; a binding that is not reached, in a handler that
   is not applied, changes nothing.

   The removal is from the term before only, and that is why it needs
   no condition: removing a binding from the term before can only take
   away a reason for it to be undefined. Removing an unused static
   binding from the term after is the converse, and is not right from
   the imports as it stands: a term after that binds an imported
   symbol, and does not use it, is undefined from the imports, and is
   the same as the term before once the binding is removed. A verdict
   that rests on such a removal is not one of the three. It holds
   under the condition that this asks for: the static environment
   binds none of the names that the groups at the head of the term
   after bind (FlambdaUnusedHeadStaticsSpec.head_free_in, the premise
   of FlambdaUnusedHeadStaticsSpec.drop_head_statics_refines_from_statement).
   [holds_from] is [holds_from_imports] with that verdict taken under
   that test, and [validate_unit_sound_head_from_statement], the same
   of the second asking, and the two at the static environment of the
   imports, are the statements of the section above with [holds_from]
   in the place of [holds_from_imports]. The test is of the term that
   the validation removes the bindings from: the term after,
   normalised and tidied.

   [validate_unit_knowing_sound_imports_statement] is the same of the
   second asking (FlambdaValidateKnowingSpec.validate_unit_knowing),
   which is [validate_unit] of the two terms normalised with what is
   known of their names (the aliases of variables, what symbols are
   bound to, the closures that a [Let] binds): those normalisations
   are under all environments too.

   [validate_unit_sound_from_statement] and
   [validate_unit_knowing_sound_from_statement] are the same two from
   any static environment in whose code and in whose values the names
   that the validation removes from the term before do not occur
   (FlambdaUnusedStaticsSpec.statics_free_of; which symbols and code
   ids the environment binds is not looked at), and from any heap in
   no cell of which they occur
   (FlambdaUnusedStaticsSpec.heap_free_of). The two above are these at
   the static environment of the imports, which holds no code and
   binds each symbol to something that holds no value, and at the
   empty heap. They are of a unit run after other units have
   bound code and symbols, as far as the model can say what those are:
   a call of code that the unit does not bind has there the meaning of
   the code that the environment binds. They say nothing of a call
   whose code is not code of the model.

   What it is not of:
   - a unit validated once unused static bindings are removed from the
     head of the term after as well
     ([Same_once_head_statics_removed]), where the static environment
     binds a name that the groups at the head of that term bind: that
     removal is the one of the paragraph above;
   - a unit validated once something is lifted
     ([Same_once_lifted]), run from a static environment in general:
     no statement here is of that verdict from a static environment
     that is not that of the imports. The statements used for it are
     those of the four liftings (FlambdaLiftFromArmsSpec,
     FlambdaLiftFromCodeSpec, FlambdaLiftSpec, FlambdaLiftAroundSpec),
     of the tables that a lifting puts at the head of the unit
     (FlambdaUnusedHeadStaticsSpec.add_head_tables) and of what is
     done to a lifted term (FlambdaKnownSymbolsSpec,
     FlambdaKnownSlotsSpec, FlambdaSharedReadsSpec,
     FlambdaUnusedSlotsSpec,
     FlambdaHoistCodeSpec, FlambdaHoistStaticsSpec), from the static
     environment of the imports. From the imports that verdict is the
     last section of this file;
   - the machine with a stack of handlers (FlambdaTrapSpec.v). Its
     agreement with this one is stated from any static environment
     that holds no code, of which that of the imports is one
     (FlambdaTrapSpec.stack_agrees_from_statement), for a unit whose
     binders are distinct and that follows the discipline of trap
     actions; no statement here puts the two together.

   [validate_unit_sound_verdict_imports_statement], and the same of
   the second asking, are of every verdict, the one that lifts among
   them, at the static environment of the imports. The law of each
   pass that the validation of a lifted term makes is stated from the
   imports in the file of the pass. Those of the rounds, of the
   removal of value slots, of the code taken first, of the reading
   of value slots that are known and of the sharing of the reads of
   two places that hold one variable ask nothing there. Those of the four
   liftings ask that the imports bind no name that the term lifted
   binds; those of the use of what the unit binds its symbols to and
   of the static bindings taken first ask it of the term that the
   pass is given (FlambdaImportsSpec.static_names_free_in). A unit
   may bind a symbol that is also an import, and the validation is
   not given the imports, so nothing in a verdict says that these
   hold: [lifted_holds_from] computes again the terms that
   [validate_unit] compares and tests each of them where the pass is
   made, with the test of the head of the term after where the
   checks of the removal of bindings from it pass, whether a binding
   goes or not. The law of the tables put at the head of the unit
   (FlambdaUnusedHeadStaticsSpec.add_head_tables, the last step of a
   lifting) asks that the imports bind no name that the groups at the
   head of its result bind: it is tested where a lifting is given a
   table, whether the table is put or not. The tests are sufficient,
   not necessary.
   [verdict_holds_from] is that test for the verdict that lifts and
   [holds_from] for the others; what is stated of it is at the static
   environment of the imports, and of the verdict that the validator
   gives of the same arguments.

   The proofs are in FlambdaValidateImportsProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaSemanticsSpec
  FlambdaCompareSpec FlambdaStaticsOrderSpec
  FlambdaUnusedStaticsSpec FlambdaUnusedHeadStaticsSpec FlambdaLoopifySpec
  FlambdaStubCallsSpec FlambdaNeverReturnsSpec
  FlambdaLiftSpec FlambdaLiftFromCodeSpec FlambdaLiftFromArmsSpec
  FlambdaLiftAroundSpec FlambdaKnownSymbolsSpec FlambdaKnownClosuresSpec
  FlambdaKnownSlotsSpec FlambdaSharedReadsSpec FlambdaHoistCodeSpec
  FlambdaHoistStaticsSpec
  FlambdaFloatCodeSpec FlambdaGuardScopeSpec
  FlambdaValidateSpec FlambdaValidateKnowingSpec
  FlambdaImportsSpec.
From OxCamlFlambda2Types Require FlambdaSimplifyCarriedSpec.
Import ListNotations.

(* Whether a verdict of [validate_unit] is one that holds of the unit
   run from its imports: every way of being validated but the one that
   removes static bindings from the head of the term after and the one
   that lifts. *)
Definition holds_from_imports (v : validation) : bool :=
  match v with
  | Same_up_to_bound_names | Same_once_normalised
  | Same_once_unused_statics_removed =>
    true
  | Same_once_head_statics_removed | Same_once_lifted | Not_validated =>
    false
  end.

(* A unit that is validated in one of the three ways that
   [holds_from_imports] accepts refines the unit it came from, each run
   on its own from the same imports. *)
Definition validate_unit_sound_imports_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (rounds : nat)
    (unused : dropped) (slots : list value_slot) (G : loopified)
    (Ls : list lifting) (kept : list code_id) (before after : expr),
    holds_from_imports
      (validate_unit rounds unused slots G Ls kept before after)
    = true ->
    unit_refines imports before after.

(* The same of the second asking. *)
Definition validate_unit_knowing_sound_imports_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (rounds : nat)
    (unused : dropped) (slots : list value_slot) (G : loopified)
    (Ls : list lifting) (kept : list code_id) (before after : expr),
    holds_from_imports
      (validate_unit_knowing rounds unused slots G Ls kept before after)
    = true ->
    unit_refines imports before after.

(* The same from any static environment in whose code and in whose
   values the names that the validation removes from the term before
   do not occur (FlambdaUnusedStaticsSpec.statics_free_of), and from
   any heap in no value of a cell of which they occur
   (FlambdaUnusedStaticsSpec.heap_free_of). *)
(* The heap is asked what the static environment is asked, as at
   FlambdaUnusedStaticsSpec.drop_statics_refines_statement: a step
   reads the heap, and a symbol that only the heap names is otherwise
   removed from under a read. The empty heap holds no value. *)
Definition validate_unit_sound_from_statement : Prop :=
  ∀ (statics : static_env) (hp : heap) (rd : read_record) (rounds : nat)
    (unused : dropped)
    (slots : list value_slot) (G : loopified) (Ls : list lifting)
    (kept : list code_id) (before after : expr),
    holds_from_imports
      (validate_unit rounds unused slots G Ls kept before after)
    = true ->
    statics_free_of unused statics = true ->
    heap_free_of unused hp = true ->
    refines (State before [] [] statics hp rd)
      (State after [] [] statics hp rd).

(* The same of the second asking. *)
Definition validate_unit_knowing_sound_from_statement : Prop :=
  ∀ (statics : static_env) (hp : heap) (rd : read_record) (rounds : nat)
    (unused : dropped)
    (slots : list value_slot) (G : loopified) (Ls : list lifting)
    (kept : list code_id) (before after : expr),
    holds_from_imports
      (validate_unit_knowing rounds unused slots G Ls kept before after)
    = true ->
    statics_free_of unused statics = true ->
    heap_free_of unused hp = true ->
    refines (State before [] [] statics hp rd)
      (State after [] [] statics hp rd).

(* ======================================================================== *)
(* The verdict that removes static bindings from the head of the term after *)
(* ======================================================================== *)

(* Whether a verdict of [validate_unit] is one that holds of the unit
   run from the static environment [statics]: the three of
   [holds_from_imports], and the one that removes static bindings from
   the head of the term after, where [statics] binds no symbol and no
   code id that the groups at the head of that term bind
   (FlambdaUnusedHeadStaticsSpec.head_free_in). The term is the one
   that [validate_unit] removes them from: [after], normalised and
   tidied. For a static environment in general the statements below
   also ask that the names removed occur in none of its code and
   values ([statics_free_of]) and in no value of a cell of the heap
   ([heap_free_of]); this test checks neither. *)
Definition holds_from (statics : static_env) (rounds : nat)
    (kept : list code_id) (after : expr) (v : validation) : bool :=
  match v with
  | Same_up_to_bound_names | Same_once_normalised
  | Same_once_unused_statics_removed =>
    true
  | Same_once_head_statics_removed =>
    head_free_in statics
      (tidied_after rounds (normalise_after rounds kept after))
  | Same_once_lifted | Not_validated => false
  end.

(* A static environment that binds no name that a term binds
   (FlambdaImportsSpec.static_names_free_in) binds none that the
   groups at its head bind. *)
Definition static_names_free_in_head_statement : Prop :=
  ∀ (statics : static_env) (e : expr),
    static_names_free_in statics e = true -> head_free_in statics e = true.

(* A unit that is validated in one of the ways that [holds_from]
   accepts refines the unit it came from, each run from a static
   environment in whose code and in whose values the names removed do
   not occur, and from a heap in no value of a cell of which they
   occur. *)
(* The heap is asked that as at [validate_unit_sound_from_statement],
   and at
   FlambdaUnusedHeadStaticsSpec.drop_head_statics_refines_from_statement. *)
Definition validate_unit_sound_head_from_statement : Prop :=
  ∀ (statics : static_env) (hp : heap) (rd : read_record) (rounds : nat)
    (unused : dropped)
    (slots : list value_slot) (G : loopified) (Ls : list lifting)
    (kept : list code_id) (before after : expr),
    holds_from statics rounds kept after
      (validate_unit rounds unused slots G Ls kept before after)
    = true ->
    statics_free_of unused statics = true ->
    heap_free_of unused hp = true ->
    refines (State before [] [] statics hp rd)
      (State after [] [] statics hp rd).

(* The same of the second asking. The term after that it gives
   [validate_unit] is the term after normalised with what is known of
   its names. *)
Definition validate_unit_knowing_sound_head_from_statement : Prop :=
  ∀ (statics : static_env) (hp : heap) (rd : read_record) (rounds : nat)
    (unused : dropped)
    (slots : list value_slot) (G : loopified) (Ls : list lifting)
    (kept : list code_id) (before after : expr),
    holds_from statics rounds kept
      (normalise_knowing (normalise_after_round kept) rounds after)
      (validate_unit_knowing rounds unused slots G Ls kept before after)
    = true ->
    statics_free_of unused statics = true ->
    heap_free_of unused hp = true ->
    refines (State before [] [] statics hp rd)
      (State after [] [] statics hp rd).

(* The two at the static environment of the imports: the test is that
   the groups at the head of the term after bind no imported symbol. *)
Definition validate_unit_sound_head_imports_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (rounds : nat)
    (unused : dropped) (slots : list value_slot) (G : loopified)
    (Ls : list lifting) (kept : list code_id) (before after : expr),
    holds_from (imports_statics imports) rounds kept after
      (validate_unit rounds unused slots G Ls kept before after)
    = true ->
    unit_refines imports before after.

Definition validate_unit_knowing_sound_head_imports_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (rounds : nat)
    (unused : dropped) (slots : list value_slot) (G : loopified)
    (Ls : list lifting) (kept : list code_id) (before after : expr),
    holds_from (imports_statics imports) rounds kept
      (normalise_knowing (normalise_after_round kept) rounds after)
      (validate_unit_knowing rounds unused slots G Ls kept before after)
    = true ->
    unit_refines imports before after.

(* ======================================================================== *)
(* The verdict that lifts                                                   *)
(* ======================================================================== *)

(* Where a step of [with_lifting] is made, whether the static
   environment binds no name that its result binds
   (FlambdaImportsSpec.static_names_free_in): the premise of the law
   of each of the four liftings from the imports. *)
Definition made_free_in (statics : static_env) (made : option expr) : bool :=
  match made with
  | Some e' => static_names_free_in statics e'
  | None => true
  end.

(* Where a lifting is given a table to put at the head of the unit
   (FlambdaUnusedHeadStaticsSpec.add_head_tables, the last step of
   [with_lifting]), whether the static environment binds no name that
   the groups at the head of the result of that step bind
   (FlambdaUnusedHeadStaticsSpec.head_free_in): the premise of
   FlambdaUnusedHeadStaticsSpec.add_head_tables_refines_imports_statement.
   It is asked whether the checks of the step pass or not; where they
   do not, the result is the term that the step was given. *)
Definition tables_free_in (statics : static_env) (L : lifting) (e : expr)
    : bool :=
  match tables_put L with
  | [] => true
  | _ :: _ => head_free_in statics (add_head_tables (tables_put L) e)
  end.

(* That, of each of the four liftings of [with_lifting L e], and the
   test of the tables. The terms are those of [with_lifting]. The
   copies of code that end [with_lifting] ask nothing of the
   environment
   (FlambdaCopyCodeSpec.with_copies_refines_imports_statement). *)
Definition lifting_free_in (statics : static_env) (L : lifting) (e : expr)
    : bool :=
  let to_head := lift_to_head_unit (lifting_unit L) (lifted_to_head L) e in
  let e :=
    match to_head with
    | Some e' => e'
    | None => e
    end
  in
  let from_code :=
    lift_from_code_unit (lifting_unit L) (lifted_out_of_code L) e
  in
  let e :=
    match from_code with
    | Some e' => e'
    | None => e
    end
  in
  let where_bound := lift_unit L e in
  let e :=
    match where_bound with
    | Some e' => e'
    | None => e
    end
  in
  let around :=
    lift_around_unit (lifting_unit L) (lifted_around L) (groups_put L) e
  in
  let e :=
    match around with
    | Some e' => e'
    | None => e
    end
  in
  made_free_in statics to_head
  && made_free_in statics from_code
  && made_free_in statics where_bound
  && made_free_in statics around
  && tables_free_in statics L e.

(* Whether, in each round of [normalise_lifted] that uses what the
   unit binds its symbols to ([with_known], where [fold_known_unit]
   gives a term), the static environment binds no name that the term
   given to [fold_known_unit] binds: the premise of
   FlambdaKnownSymbolsSpec.fold_known_unit_refines_imports_statement.
   The terms are those of [normalise_lifted_round], and the rounds
   are those of [iterate]: the test stops where [iterate] stops, after
   the first round that gives back the term it was given, up to bound
   names. *)
Fixpoint known_free_in (statics : static_env) (rounds : nat)
    (avoid : list continuation) (kept : list code_id) (e : expr) : bool :=
  match rounds with
  | O => true
  | S rounds =>
    let given :=
      use_known_closures (normalise_before_round_avoiding avoid kept e)
    in
    let rest (e' : expr) : bool :=
      if expr_eqb [] [] e e' then true
      else known_free_in statics rounds avoid kept e'
    in
    match fold_known_unit given with
    | Some folded =>
      static_names_free_in statics given
      && rest (share_static_reads (read_known_slots folded))
    | None => rest (share_static_reads (read_known_slots given))
    end
  end.

(* That, of the rounds of [normalise_lifted rounds avoid kept e]:
   those before the rewrites that Simplify makes from the types, and
   those after. *)
Definition normalise_lifted_free_in (statics : static_env) (rounds : nat)
    (avoid : list continuation) (kept : list code_id) (e : expr) : bool :=
  let e := iterate rounds first_phase_round e in
  known_free_in statics rounds avoid kept e
  && known_free_in statics rounds avoid kept
       (guard_scope FlambdaSimplifyCarriedSpec.simplify_by_types_carried
          (iterate rounds (normalise_lifted_round avoid kept) e)).

(* [known_free_in] of the rounds of
   FlambdaValidateSpec.normalise_lifted_calling: of each round, the
   test of one round of [normalise_lifted] ([known_free_in] of one
   round), asked of the term that the round is given; the step that
   follows the round, where [calling] is set, asks nothing, being a
   refinement under all environments. With [calling] unset it is
   [known_free_in] ([known_calling_free_in_unset_statement]). *)
Fixpoint known_calling_free_in (statics : static_env) (calling : bool)
    (rounds : nat) (avoid : list continuation) (kept : list code_id)
    (e : expr) : bool :=
  match rounds with
  | O => true
  | S rounds =>
    known_free_in statics 1 avoid kept e
    && (let e' := normalise_lifted_round_calling calling avoid kept e in
        if expr_eqb [] [] e e' then true
        else known_calling_free_in statics calling rounds avoid kept e')
  end.

Definition known_calling_free_in_unset_statement : Prop :=
  ∀ (statics : static_env) (rounds : nat) (avoid : list continuation)
    (kept : list code_id) (e : expr),
    known_calling_free_in statics false rounds avoid kept e
    = known_free_in statics rounds avoid kept e.

(* [normalise_lifted_free_in] of the rounds of
   [normalise_lifted_calling calling rounds avoid kept e]. With
   [calling] unset it is [normalise_lifted_free_in]
   ([normalise_lifted_calling_free_in_unset_statement]). *)
Definition normalise_lifted_calling_free_in (statics : static_env)
    (calling : bool) (rounds : nat) (avoid : list continuation)
    (kept : list code_id) (e : expr) : bool :=
  let e := iterate rounds first_phase_round e in
  known_calling_free_in statics calling rounds avoid kept e
  && known_calling_free_in statics calling rounds avoid kept
       (guard_scope FlambdaSimplifyCarriedSpec.simplify_by_types_carried
          (iterate rounds (normalise_lifted_round_calling calling avoid kept)
             e)).

Definition normalise_lifted_calling_free_in_unset_statement : Prop :=
  ∀ (statics : static_env) (rounds : nat) (avoid : list continuation)
    (kept : list code_id) (e : expr),
    normalise_lifted_calling_free_in statics false rounds avoid kept e
    = normalise_lifted_free_in statics rounds avoid kept e.

(* [known_free_in] of the rounds of
   FlambdaValidateSpec.normalise_lifted_own: of each round, the test
   of one round of [normalise_lifted], asked of the term that the
   round is given, as in [known_calling_free_in]; the step that ends
   the round where [own] is set
   (FlambdaClosureOfCodeSpec.know_own_closures) asks nothing, its law
   from the imports having no premise. With [own] unset it is
   [known_calling_free_in] ([known_own_free_in_unset_statement]). *)
Fixpoint known_own_free_in (statics : static_env) (calling own : bool)
    (rounds : nat) (avoid : list continuation) (kept : list code_id)
    (e : expr) : bool :=
  match rounds with
  | O => true
  | S rounds =>
    known_free_in statics 1 avoid kept e
    && (let e' := normalise_lifted_round_own calling own avoid kept e in
        if expr_eqb [] [] e e' then true
        else known_own_free_in statics calling own rounds avoid kept e')
  end.

Definition known_own_free_in_unset_statement : Prop :=
  ∀ (statics : static_env) (calling : bool) (rounds : nat)
    (avoid : list continuation) (kept : list code_id) (e : expr),
    known_own_free_in statics calling false rounds avoid kept e
    = known_calling_free_in statics calling rounds avoid kept e.

(* [normalise_lifted_free_in] of the rounds of
   [normalise_lifted_own calling own rounds avoid kept e]. With [own]
   unset it is [normalise_lifted_calling_free_in]
   ([normalise_lifted_own_free_in_unset_statement]). *)
Definition normalise_lifted_own_free_in (statics : static_env)
    (calling own : bool) (rounds : nat) (avoid : list continuation)
    (kept : list code_id) (e : expr) : bool :=
  let e := iterate rounds first_phase_round e in
  known_own_free_in statics calling own rounds avoid kept e
  && known_own_free_in statics calling own rounds avoid kept
       (guard_scope FlambdaSimplifyCarriedSpec.simplify_by_types_carried
          (iterate rounds (normalise_lifted_round_own calling own avoid kept)
             e)).

Definition normalise_lifted_own_free_in_unset_statement : Prop :=
  ∀ (statics : static_env) (calling : bool) (rounds : nat)
    (avoid : list continuation) (kept : list code_id) (e : expr),
    normalise_lifted_own_free_in statics calling false rounds avoid kept e
    = normalise_lifted_calling_free_in statics calling rounds avoid kept e.

(* The same test in each round of
   FlambdaValidateSpec.demote_and_tidy_known that uses what the unit
   binds its symbols to. The terms are those of
   [demote_and_tidy_known_round], with [inner] the number of rounds
   that it gives [demote_and_tidy], and the rounds are those of
   [iterate]: the test stops where [iterate] stops. *)
Fixpoint tidy_known_free_in (statics : static_env) (inner rounds : nat)
    (e : expr) : bool :=
  match rounds with
  | O => true
  | S rounds =>
    let given := use_known_closures e in
    let rest (e' : expr) : bool :=
      if expr_eqb [] [] e e' then true
      else tidy_known_free_in statics inner rounds e'
    in
    match fold_known_unit given with
    | Some folded =>
      static_names_free_in statics given
      && rest
           (demote_and_tidy inner
              (share_static_reads (read_known_slots folded)))
    | None =>
      rest
        (demote_and_tidy inner (share_static_reads (read_known_slots given)))
    end
  end.

Definition demote_and_tidy_known_free_in (statics : static_env)
    (rounds : nat) (e : expr) : bool :=
  tidy_known_free_in statics rounds rounds (demote_and_tidy rounds e).

(* The test of the lifting and that of the normalisation that follows
   it ([normalise_lifted_own_free_in], with what the lifting says of
   the calls of the closures that a [Let] makes and of what
   [my_closure] is known to be), of each of the
   liftings as [lifted_in_turn] applies them, and after the last that
   of what ends the normalisation, asked of the term that the
   normalisation ends: the term with its known value slots read with
   every symbol in scope
   (FlambdaKnownSlotsSpec.read_known_slots_anywhere), for which
   nothing is asked, as its law from the imports has no premise. *)
Fixpoint lifted_free_in (statics : static_env) (rounds : nat)
    (avoid : list continuation) (Ls : list lifting) (kept : list code_id)
    (e : expr) : bool :=
  match Ls with
  | [] =>
    demote_and_tidy_known_free_in statics rounds (read_known_slots_anywhere e)
  | L :: Ls =>
    lifting_free_in statics L e
    && normalise_lifted_own_free_in statics (made_closures_called L)
         (own_closures_known L) rounds avoid kept (with_lifting L e)
    && lifted_free_in statics rounds avoid Ls kept
         (normalise_lifted_own (made_closures_called L)
            (own_closures_known L) rounds avoid kept (with_lifting L e))
  end.

(* Where [with_statics_first Ls e] takes bindings to the head of [e],
   whether the static environment binds no name that [e] binds: the
   premise of
   FlambdaHoistStaticsSpec.hoist_statics_equivalent_imports_statement. *)
Definition statics_first_free_in (statics : static_env) (Ls : list lifting)
    (e : expr) : bool :=
  match Ls with
  | [] => true
  | L :: _ =>
    match hoist_statics (lifting_unit L) e with
    | Some _ => static_names_free_in statics e
    | None => true
    end
  end.

(* Where the removal of static bindings from the head of [e] is made
   ([drop_head_statics D e] gives a term: its checks pass, whether or
   not a binding goes), whether the static environment binds no name
   that the groups at the head of [e] bind: the premise of
   FlambdaUnusedHeadStaticsSpec.drop_head_statics_refines_from_statement. *)
Definition head_removal_free_in (statics : static_env) (D : dropped)
    (e : expr) : bool :=
  match drop_head_statics D e with
  | Some _ => head_free_in statics e
  | None => true
  end.

(* Whether a verdict [Same_once_lifted] of [validate_unit], of the
   arguments given, is one that holds of the unit run from the static
   environment [statics]. The terms are those that [validate_unit]
   compares, computed again. Asked:
   - of the term after, normalised and tidied,
     [head_removal_free_in];
   - of the term before, [lifted_free_in];
   - where the first of the two comparisons of lifted terms fails, so
     that the verdict is of the second, [statics_first_free_in] of
     both terms that the second compares.
   Nothing is asked for the removal of value slots, for the code taken
   first, for the reading of value slots that are known, or for the
   sharing of the reads of two places that hold one variable: their laws
   from the imports have no premise. Nor for the calls of the lifted
   term that are taken on when its rounds are over
   (FlambdaValidateSpec.with_returns, under the choice that
   [validate_unit] makes from the term after): that is a refinement
   under all environments. The tests are sufficient, and no
   more is claimed of them: a unit that fails one may be correct from
   its imports. What is stated of this function is the statement
   below, at the static environment of the imports and of arguments
   of which [validate_unit] gives that verdict; nothing is stated of
   it at another static environment, nor of arguments of which the
   verdict is another. *)
Definition lifted_holds_from (statics : static_env) (rounds : nat)
    (unused : dropped) (slots : list value_slot) (G : loopified)
    (Ls : list lifting) (kept : list code_id) (before after : expr) : bool :=
  let before :=
    loopify_unit G (cut_returns [] (demote_full_stub_calls before))
  in
  let tidied := tidied_after rounds (normalise_after rounds kept after) in
  let taken := negb (keeps_a_return tidied) in
  let after := without_head_statics unused tidied in
  let lifted :=
    without_statics unused
      (without_slots slots
         (tidied_before rounds
            (with_returns taken
               (lifted_in_turn rounds (unit_continuations Ls) Ls kept
                  before))))
  in
  head_removal_free_in statics unused tidied
  && lifted_free_in statics rounds (unit_continuations Ls) Ls kept before
  && (expr_eqb [] []
        (order_statics_as (float_code (with_code_first after))
           (float_code lifted))
        (float_code (with_code_first after))
      || statics_first_free_in statics Ls after
         && statics_first_free_in statics Ls lifted).

(* Whether a verdict of [validate_unit], of the arguments given, is one
   that holds of the unit run from the static environment [statics]:
   [lifted_holds_from] for the verdict that lifts, [holds_from] for
   the others. The verdict is to be the one that [validate_unit] gives
   of those arguments: the function does not compute it, and the
   statement below is of that one alone, at the static environment of
   the imports. *)
Definition verdict_holds_from (statics : static_env) (rounds : nat)
    (unused : dropped) (slots : list value_slot) (G : loopified)
    (Ls : list lifting) (kept : list code_id) (before after : expr)
    (v : validation) : bool :=
  match v with
  | Same_once_lifted =>
    lifted_holds_from statics rounds unused slots G Ls kept before after
  | Same_up_to_bound_names | Same_once_normalised
  | Same_once_unused_statics_removed | Same_once_head_statics_removed
  | Not_validated =>
    holds_from statics rounds kept after v
  end.

(* The same of a verdict of the second asking, of the arguments that
   [validate_unit_knowing] is given: the test is of the two terms that
   it gives [validate_unit]. The verdict is to be the one that
   [validate_unit_knowing] gives of those arguments. *)
Definition verdict_knowing_holds_from (statics : static_env) (rounds : nat)
    (unused : dropped) (slots : list value_slot) (G : loopified)
    (Ls : list lifting) (kept : list code_id) (before after : expr)
    (v : validation) : bool :=
  verdict_holds_from statics rounds unused slots G Ls kept
    (normalise_knowing
       (normalise_before_round_avoiding
          (unit_continuations Ls ++ map snd G) kept)
       rounds before)
    (normalise_knowing (normalise_after_round kept) rounds after)
    v.

(* A unit whose verdict [verdict_holds_from] accepts at the static
   environment of its imports refines the unit it came from, each run
   on its own from those imports. *)
Definition validate_unit_sound_verdict_imports_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (rounds : nat)
    (unused : dropped) (slots : list value_slot) (G : loopified)
    (Ls : list lifting) (kept : list code_id) (before after : expr),
    verdict_holds_from (imports_statics imports) rounds unused slots G Ls
      kept before after
      (validate_unit rounds unused slots G Ls kept before after)
    = true ->
    unit_refines imports before after.

(* The same of the second asking. *)
Definition validate_unit_knowing_sound_verdict_imports_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (rounds : nat)
    (unused : dropped) (slots : list value_slot) (G : loopified)
    (Ls : list lifting) (kept : list code_id) (before after : expr),
    verdict_knowing_holds_from (imports_statics imports) rounds unused slots
      G Ls kept before after
      (validate_unit_knowing rounds unused slots G Ls kept before after)
    = true ->
    unit_refines imports before after.

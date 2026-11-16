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

(* Validating what Simplify did to a compilation unit.

   This file is to be reviewed: [validate_unit] and the two statements.
   It is extracted and run by the test of
   middle_end/flambda2/tests/semantics/ on the term of a unit before
   Simplify and the term after.

   [validate_unit] is the whole check, so that what it accepts is correct
   by one theorem and not by how a tool puts the proved pieces together.
   It compares the two terms as they are given, up to bound names
   (FlambdaCompareSpec.expr_eqb). Failing that, it makes indirect, in
   the code of each stub of the term before, the direct calls of code
   of the unit that is not a stub, as Simplify does where it
   simplifies the code of a stub
   (FlambdaStubCallsSpec.demote_stub_calls); it puts each direct call
   of code in sight whose body does not mention its return
   continuation once the bindings of the continuations that nothing
   uses are removed from it, where the other checks of the rewrite pass
   (FlambdaNeverReturnsSpec.never_returns_b and [invalid_params]: the
   call is not a partial application, the code is not tupled, the two
   continuations of the call are two names, and the code binds as
   many names as the call has results), under a
   binding of the call's own return continuation to a handler that is
   [Invalid], which is the form that the term after has for a call to
   which Simplify gives no return continuation
   (FlambdaNeverReturnsSpec.cut_returns; after the demotion in the
   code of stubs, so that a direct call there of code of the unit
   that is not a stub is indirect by then and is left: at the default
   flags Simplify keeps the return continuation of a call, in the
   code of a stub, whose callee is projected from the closure of the
   stub); and it makes jumps of the
   self tail calls of the code named, each with the name given for its
   loop (FlambdaLoopifySpec.loopify_unit, which says what a self tail
   call is and leaves alone the code that it may not change), as
   Simplify does for the code that the front end marks; the result is
   the term before for all that follows. It normalises that term by
   rewrites that refine it
   ([normalise_before]) and the term after by rewrites that are
   equivalences ([normalise_after]), each gone round a number of times as
   they enable one another, and compares the results up to bound
   names. The term before is normalised in two
   phases: first what Simplify does to the parameters of the
   continuations of the term as it is ([normalise_params_round]; in
   the code and in the loops of the term, the unboxing of a parameter
   knows the static blocks and boxed numbers of the static bindings in
   scope, FlambdaUnboxInScopeSpec.v), each round followed by the
   going of applications to the arms of a handler that tests what
   they give ([first_phase_round]), then
   the rounds that inline the calls of known code
   ([normalise_before_round]), which the term after has too. Failing
   that, it removes from the term
   before the static bindings of the names given
   (FlambdaUnusedStaticsSpec.drop_statics) and compares again. Failing
   that, it removes from the term after too those of them that the
   term after binds in the groups at its head, where the checks of
   FlambdaUnusedHeadStaticsSpec.head_statics_removable pass
   ([without_head_statics]), and compares again; a pair found the same
   so has a verdict of its own ([Same_once_head_statics_removed]).
   Simplify leaves such bindings where it left a call that the
   normalisation has since inlined in both terms: the code that was
   called and its closure. The term after so cleaned is the term
   after of the comparisons that follow. That the term after refines
   it is a law of a unit run on its own from empty environments, and
   not one under all environments nor from a static environment that
   binds something already (FlambdaUnusedHeadStaticsSpec.v says why):
   so it comes after the comparisons whose verdicts
   [validate_unit_sound_everywhere_statement] speaks of, and after
   the comparison that removes from the term before only, and none
   of those three verdicts rests on it. Failing
   that too, it starts again from the term before as it was given. For
   each of the liftings given, in turn ([lifted_in_turn]), it makes the
   [Let]s given static bindings of the symbols given: first those to
   bind at the head of the unit, which mention no variable and may
   stand where no static binding can, in the arm of a test
   (FlambdaLiftFromArmsSpec.lift_to_head_unit); then those in
   the bodies of code, which mention no variable and whose bindings go
   beside the code (FlambdaLiftFromCodeSpec.lift_from_code_unit); then
   those outside code, where they stand (FlambdaLiftSpec.lift_unit,
   which says what is lifted and under which conditions). With each
   lifting it also binds, at the head of the unit, the tables given:
   symbols bound to immutable arrays of constants and symbols, which
   Simplify binds where it makes a load of a [Switch] whose arms give
   constants, and which the term before has not
   (FlambdaUnusedHeadStaticsSpec.add_head_tables). Then it gives the
   static sets of closures of the symbols given the copies of code
   given: where Simplify has given a set that it lifts new code,
   beside the code of the term before, which another closure keeps,
   the term before gets the same code under the new code ids, for
   that set (FlambdaCopyCodeSpec.copy_static_code, which says what
   Simplify does there and what is checked). It
   normalises the result, each round followed by using
   what is known of the static closures whose bindings are in sight
   (FlambdaKnownClosuresSpec.use_known_closures: a value slot is read,
   a call is made direct), by replacing the primitives and the
   projections applied to symbols that the unit binds to constants by
   their values (FlambdaKnownSymbolsSpec.fold_known_unit), and by
   replacing the projection of a value slot that every set of
   closures of the unit gives the same symbol or constant by that
   symbol or constant, where the symbol is in scope and the code is
   not that of a stub (FlambdaKnownSlotsSpec.read_known_slots): in
   the code of a closure that a function makes, the slot that holds a
   function of the unit is the symbol of that function once the
   function is lifted; and by making the read of a place of a static
   constant the variable of an earlier read, where that variable is
   still in scope and the two places are of the constants of one
   group of static bindings, which gave them one variable
   (FlambdaSharedReadsSpec.share_static_reads; [normalise_lifted]):
   where the code of a function reads a value that is computed when
   the unit runs, once through its own closure and once through the
   block of its module, the second read is the variable of the first.
   Between those rounds and the same rounds once
   more it applies the rewrites from the types, as the normalisation
   of the term before does. Where the lifting says so, each of those
   rounds also makes direct the calls of the closures that a [Let]
   makes (FlambdaClosureCallsSpec.call_made_closures;
   [normalise_lifted_calling]); where the lifting does not say so,
   the rounds are those that there were. Where the lifting says so,
   each of those rounds also uses, in the bodies of code, what
   [my_closure] is known to be from its code
   (FlambdaClosureOfCodeSpec.know_own_closures;
   [normalise_lifted_own]); where it does not, the rounds are again
   those that there were. Every round of that
   normalisation, after each
   of the liftings, makes no name that is the return or the exception
   continuation of the unit of any of the liftings given, the later
   ones included, whether or not the term mentions them
   ([unit_continuations]; FlambdaRenameApartSpec.rename_apart_avoiding
   takes its new names outside the free names of the term and outside
   those; a new name may be one that the term binds): a lifting checks
   the unit by the names of those two
   continuations, which the term may no longer mention once it is
   normalised, so a binder made in a round must not take one of them.
   After the last of the liftings it reads the known value slots once
   more, with every symbol that the unit binds taken to be in scope
   (FlambdaKnownSlotsSpec.read_known_slots_anywhere): in code that
   the text of the unit binds above the symbol too, where Simplify
   reads the slot and row 4 would not let it be read. No lifting
   comes after, and what comes after and tests row 4
   (FlambdaHoistStaticsSpec.hoist_statics, in [with_statics_first])
   tests the term that it is given and leaves it as it is if the
   test fails.
   Then it removes from the static sets of closures the value slots
   given, if nothing projects them
   (FlambdaUnusedSlotsSpec.drop_unused_slots), and after them the
   [Let]s that this leaves unused, in the term and in the bodies of
   its code ([without_slots]); removes the static bindings of the
   names given, and compares; in this comparison the
   bindings of code of both terms are first moved up to the head of
   what they are in (FlambdaFloatCodeSpec.float_code), as the two terms
   do not bind the code of a lifted closure in the same place; and
   before that the term after, normalised, has all its code bound at
   its head, where it passes the checks of
   FlambdaHoistCodeSpec.hoist_code ([with_code_first]): Simplify binds
   the code of a closure that it lifts in the handler of a
   continuation in that handler, and the term before binds all the
   code of the unit first. Failing that, it compares the same two
   terms otherwise: each with its static bindings that need nothing
   from around them, the code and the symbols whose constants mention
   no variable, bound first, where it passes the checks of
   FlambdaHoistStaticsSpec.hoist_statics for the unit of the first
   lifting given ([with_statics_first]). Simplify binds a symbol that
   it lifts out of the arm of a test under the last [Let] before the
   test, and the term before binds it at its head; where such a
   binding stands does not matter to a unit whose static names are
   all in scope where they are mentioned. The
   first lifting comes before any normalisation because the
   normalisation unfolds calls: a [Let] in the body of a function would
   be many [Let]s afterwards, and a closure that nothing uses would be
   gone. A later lifting names [Let]s of the term as the liftings before
   it and the normalisation have left it: those that the inlining of a
   call has brought out of a function, which Simplify lifts where it
   inlines the call. What is known of the symbols is used in every
   round because they enable one another: a round puts a symbol where
   a variable stood for it; what is known of the symbol then puts what
   a projection of it gives in place of the projection, and the code of
   the closure in a call of it; and the next round follows the one to
   where it is read in turn, and inlines the other. The value slots
   come last because it is the normalisation that removes the
   projections which the code of a function makes and no longer needs.
   Before each of the comparisons of normalised terms the bindings of
   each group of static bindings of the term before are put in the
   order that the term after has them in
   (FlambdaStaticsOrderSpec.order_statics_as): Simplify builds the group
   of a set of closures again, in an order of its own.

   A round of the second phase of the term before that has left the
   term as it was, up to bound names, makes the twins of the
   parameters of continuations that hold a tagged immediate
   ([with_twins]): the continuation of a parameter that its handler
   untags gets one parameter more, which holds the number, and the
   handler reads the number there; the parameter stays for its other
   uses. Simplify unboxes such a parameter so, and keeps it beside
   the number where something else uses it.
   FlambdaShadowParamsSpec.v and FlambdaTwinParamsSpec.v say which
   parameters, and where the second has its name from; the comment
   at [with_twins] says why this is not done in every round.

   When the rounds of the term before are over, and before its
   continuations are placed, its calls are taken on ([with_returns]):
   a call that returns to a continuation whose handler passes its
   parameters on as they are returns to the continuation that the
   handler applies (FlambdaForwardSpec.forward_applications_with,
   with the calls), in the handlers of recursive continuations and
   in the bodies of code too, and the continuations that this leaves
   unused are removed ([returns_taken_on]). Simplify does so for a
   call (FlambdaForwardSpec.v says where). It is not done in a
   round, and it is not done at all if the term after, as it is
   compared, has a call that the same rewrite would take on
   ([keeps_a_return]). In the pair that showed it, Simplify left that
   call because the handler ended a region before it passed its
   parameters on, which the model has not got. Whatever left the
   call there, the term after cannot be given the rewrite, which is
   not an equivalence; the term before then keeps its calls where
   they return. Either way the term before is refined, so the choice
   is the validator's own and nothing is asked of it. The same is
   done to the term before with the liftings made.

   Apart from the first comparison, which is of the two terms as they
   are given, each of the terms that are compared has first had its
   continuations placed ([placed]): a non-recursive [Let_cont] goes
   down past the [Let]s and the other bindings under it that allow
   it, and into the handler of a non-recursive [Let_cont] under it
   that alone names its continuation, as far as it can
   (FlambdaSinkUsesSpec.sink_conts_to_uses, an equivalence), in the
   bodies of code and in the handlers of recursive continuations
   too. It then stands at the call that returns to its continuation,
   at the test of which an arm names it or at the loop that leaves
   by it, where nothing that it can neither pass nor enter is
   between the two. It is done once, when the rounds are
   over: to the term before normalised, to the term after normalised,
   and to the term before with the liftings made. Simplify binds a
   continuation that it makes for a call at the call, under the [Let]s
   of the arguments, where the term before has the continuation of
   the call above them; and the comparison is of the bindings in
   their order.

   Each of those terms is then tidied, in rounds ([tidied_before],
   [tidied_after]). Simplify gives a continuation a parameter for a
   primitive that was computed on every path to it, and gives each
   application the variable that holds the result
   (FlambdaUnaliasParamsSpec.v says where), and the handler uses the
   parameter; the term before computes the primitive again in the
   handler. With the continuation placed under the [Let] of the
   primitive, the variable is in scope at the handler: where the
   checks of FlambdaUnaliasParamsSpec.unalias_params pass, the
   parameter is taken away and the handler begins with a [Let] of it
   to the variable, in both terms. The same is done for a parameter
   of a loop that every turn passes on as it is and that the loop is
   given one variable for
   (FlambdaUnaliasRecParamsSpec.unalias_rec_params), in both terms,
   so that the two do not differ by such a parameter. On a path
   through an arm of a [Switch] on the result, what Simplify gives
   for the parameter is the number of the arm; such an argument is
   first made the scrutinee
   (FlambdaGiveScrutineeSpec.give_scrutinees), where that lets the
   parameter be taken away. Where it does not, the argument
   stays, and the term before has the scrutinee where Simplify has
   the number, as for a flag that a loop tests before it sets it: the
   scrutinee is made the number of the arm, in the arm and in the
   handler that only that arm applies, at the reads that
   FlambdaArmNumbersSpec.v lists (an argument of an application of a
   continuation, a defining expression that is the variable, an
   argument of a primitive, the scrutinee of a [Switch]; a call, a
   set of closures, a projection and a static binding keep the
   variable), and the variable that it is the untagging of is made
   the tagged number at the last three of those reads
   (FlambdaArmNumbersSpec.numbers_in_arms), in both terms. What that
   makes a primitive on constants or a [Switch] on a constant is
   folded, in both terms ([fold_constants]): Simplify, which
   simplifies the handler knowing the number, has the value and the
   arm. The arms of a [Switch] of the term before may then all be
   the same, and the [Switch] is that arm ([collapse_switches]), or
   all give their numbers, and such a [Switch] is the application of
   its continuation to the scrutinee, as Simplify makes it
   (FlambdaMergeSwitchesSpec.merge_switches, which the rounds apply
   before the scrutinee is a number there). What this leaves is then
   taken away as a round of the normalisation takes it, no round
   following the placing: a continuation left unused, and one left
   applied once, whose handler is put at its application
   ([remove_unused_conts], [sink_let_conts], and [inline_tail_conts]
   or, for the term after,
   FlambdaInlineTailSpec.inline_tail_conts_kinded). Then
   what the handler of the term before computes again is shared with
   the first computation ([share_known_primitives]); aliases are
   followed and the [Let]s left unused are removed. The same is done
   in the handlers of recursive continuations and in the bodies of
   code. The term after is given the sharing too: placing puts side
   by side, in both terms, computations of one primitive that a
   [Let_cont] stood between and that Simplify did not share, and the
   two terms must come out of it alike. Each rewrite of the tidying
   of the term after is an equivalence
   (FlambdaGiveScrutineeSpec.give_scrutinees_equivalent_statement,
   FlambdaUnaliasParamsSpec.unalias_params_equivalent_statement,
   FlambdaUnaliasRecParamsSpec.unalias_rec_params_equivalent_statement,
   FlambdaArmNumbersSpec.numbers_in_arms_equivalent_statement,
   FlambdaNormaliseSpec.fold_constants_equivalent_statement,
   FlambdaUnusedContsSpec.remove_unused_conts_equivalent_statement,
   FlambdaSinkContsSpec.sink_let_conts_equivalent_statement,
   FlambdaInlineTailSpec.inline_tail_conts_kinded_equivalent_statement,
   FlambdaCseEquivalentSpec.share_known_primitives_equivalent_statement).
   The tidying is not in a round of the normalisations for the reason
   that the placing is not: it needs the continuations placed, and
   the rewrites of a round that look at where a continuation is bound
   follow Simplify on the term as the rounds leave it. When its rounds
   are over, outside the bodies of code and the handlers of recursive
   continuations, a group of static bindings at the head of the body
   of a [Let_cont] is bound before the [Let_cont] instead
   (FlambdaGroupsUpSpec.groups_up, an equivalence), and groups that
   then follow one another are made one: the two terms do not bind a
   symbol in the same place among the continuations of a test, and a
   round may remove a [Let] that stood between a [Let_cont] and a
   group, or between two groups.

   The normalisation of the term before ends by making ordinary
   continuations of the exception handlers that no call can raise to,
   and removing the trap actions that name them
   (FlambdaDemoteSpec.demote_handlers), as Simplify does to a [try]
   all of whose raises it sees, and then going through its rounds
   again ([demote_and_tidy]): the term after has a jump where the term
   before has a raise, and the comparison tells a trap action from
   none. The same is done after the last of the liftings, and there
   what is known of the symbols of the unit is then used again
   ([demote_and_tidy_known]): a handler that is demoted may come to be
   applied to a symbol that a lifting bound. The rewrite
   is applied with no name in scope. It takes the names in scope so as
   to leave alone a handler whose name is also that of a continuation
   around the term, and a term that passes the checks of exception
   handlers and of trap actions then still passes them; nothing here
   asks those checks of a normalised term, and the rewrite is an
   equivalence whatever names it is given
   (FlambdaDemoteSpec.demote_handlers_equivalent_statement), which is
   all that the two statements below use. It does not look at the
   kind of a raise. On a unit compiled with debugging information
   Simplify keeps the handler that a raise with a backtrace reaches,
   and the rewrite demotes it: that can leave the two normalised terms
   different, and prevent the validation.

   What it accepts is correct for the unit run as a compilation unit is
   run, on its own from empty environments
   ([validate_unit_sound_statement]): if the term before exits, the term
   after exits alike, and if the term after terminates then it does as
   the term before does, or the term before is undefined. Where no static
   binding had to be removed and nothing lifted, this holds under all
   environments too ([validate_unit_sound_everywhere_statement]). A unit
   in which something was lifted exits with constants and symbols only:
   that is one of the conditions.

   The number of rounds, the names and the value slots to remove, the
   code to loopify with the names of its loops, what to lift and the
   code whose calls are not to be inlined
   (FlambdaNormaliseSpec.inline_calls; the same on the two sides) are
   for the caller to choose; none is trusted. Fewer rounds, the wrong
   names, the wrong value slots, the wrong loops, the wrong [Let]s or
   the wrong code make the check fail, not accept what it should not.
   The names to remove are those of the term that is compared: of the
   term lifted, where something is lifted.

   The proofs are in FlambdaValidateProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaSemanticsSpec
  FlambdaCompareSpec FlambdaNormaliseSpec FlambdaBlockLoadSpec
  FlambdaCodeLetsSpec FlambdaCseSpec FlambdaDeadParamsSpec FlambdaForwardSpec
  FlambdaForwardKindedSpec FlambdaForwardArmSpec FlambdaLiftContsSpec
  FlambdaIdentitiesSpec FlambdaMergeSwitchesSpec FlambdaMergeTablesSpec
  FlambdaInlineSiblingsSpec FlambdaInlineTailSpec FlambdaLoopAliasesSpec
  FlambdaLoopLetsSpec
  FlambdaParamAliasesSpec FlambdaRecHandlersSpec
  FlambdaRecoverComparisonSpec FlambdaRenameApartSpec
  FlambdaSameComparisonSpec
  FlambdaShareProjectionsSpec FlambdaShortcutSpec FlambdaSinkContsSpec
  FlambdaSinkUsesSpec FlambdaUnaliasParamsSpec
  FlambdaStaticsOrderSpec FlambdaUnboxSpec FlambdaUnboxVariantsSpec
  FlambdaUnusedContsSpec
  FlambdaUnusedParamsSpec FlambdaUnusedProjectionsSpec
  FlambdaReadsAfterSpec
  FlambdaUnusedStaticsSpec FlambdaWellFormedSpec FlambdaLiftSpec
  FlambdaUnusedSlotsSpec FlambdaKnownSymbolsSpec FlambdaKnownClosuresSpec
  FlambdaFloatCodeSpec FlambdaHoistCodeSpec FlambdaLoopifySpec
  FlambdaLiftFromCodeSpec FlambdaLiftFromArmsSpec FlambdaLiftAroundSpec
  FlambdaHoistStaticsSpec
  FlambdaGuardScopeSpec FlambdaGroupsUpSpec.
From OxCamlFlambda2 Require Import FlambdaWrapReturnsSpec
  FlambdaUntagParamsSpec FlambdaGiveScrutineeSpec FlambdaArmNumbersSpec
  FlambdaShadowParamsSpec FlambdaTwinParamsSpec.
From OxCamlFlambda2 Require Import FlambdaAliasLetsSpec.
From OxCamlFlambda2 Require Import FlambdaKnownSiblingsSpec.
From OxCamlFlambda2 Require Import FlambdaDemoteSpec.
From OxCamlFlambda2 Require Import FlambdaKnownSlotsSpec.
From OxCamlFlambda2 Require Import FlambdaSharedReadsSpec.
From OxCamlFlambda2 Require Import FlambdaStubCallsSpec.
From OxCamlFlambda2 Require Import FlambdaCopyCodeSpec.
From OxCamlFlambda2 Require Import FlambdaClosureCallsSpec.
From OxCamlFlambda2 Require Import FlambdaClosureOfCodeSpec.
From OxCamlFlambda2 Require Import FlambdaNeverReturnsSpec.
From OxCamlFlambda2 Require Import FlambdaUnusedHeadStaticsSpec.
From OxCamlFlambda2 Require Import FlambdaSameHandlersSpec.
From OxCamlFlambda2 Require FlambdaBindConstParamsSpec.
From OxCamlFlambda2 Require Import FlambdaUnaliasRecParamsSpec.
From OxCamlFlambda2 Require Import FlambdaMadeLengthsSpec.
From OxCamlFlambda2 Require Import FlambdaUnboxParamsSpec
  FlambdaUnboxInScopeSpec FlambdaUnboxLoopsSpec FlambdaUnboxTogetherSpec.
From OxCamlFlambda2Types Require Import FlambdaAliasesSpec.
From OxCamlFlambda2Types Require FlambdaSimplifyPassSpec.
From OxCamlFlambda2Types Require FlambdaSimplifyCarriedSpec.
Import ListNotations.

(* ======================================================================== *)
(* The normalisations                                                       *)
(* ======================================================================== *)

(* A rewrite applied [rounds] times at most. It stops after the first
   application that gives back the term it was given, up to bound
   names (FlambdaCompareSpec.expr_eqb), and the result is what that
   application gave: so, where [rounds] is not zero, the rewrite has
   been applied once at least, and the result has what the rewrite
   does to bound names (a rewrite of the validation that makes names
   gives every binder a name of its own, and later rewrites read
   that). The rounds after that one are not made. No claim is made
   that further applications would leave the term unchanged, even up
   to bound names: what is stated of [iterate] holds of any number of
   applications. *)
Fixpoint iterate (rounds : nat) (f : expr -> expr) (e : expr) : expr :=
  match rounds with
  | O => e
  | S rounds =>
    let e' := f e in
    if expr_eqb [] [] e e' then e' else iterate rounds f e'
  end.

(* One round for an expression of the term before: the rewrites that
   need to know nothing of where they are applied, and that go into the
   handlers of the non-recursive continuations but not into those of the
   recursive ones, nor into code. First what makes the control flow
   simpler: a [Switch] whose arms all apply one continuation, to
   arguments that are the same, or the number of the arm, tagged or
   not, or the negation of that number where the arms are those of 0
   and 1, or, where there are three arms or more, that number times a
   slope plus an offset, becomes that application, under the [Let]s
   that compute the argument where it is not the scrutinee itself,
   where the tests of FlambdaMergeSwitchesSpec.merge_switches pass
   (among them: the scrutinee is a variable, no arm has a trap action,
   either the arguments that differ are all computed in one way and
   no arm passes the scrutinee, or the continuation is bound around
   the [Switch] and its parameters can be bound to the arguments that
   are computed, and, for the ways that tag or compute, the numbers
   of the arms are immediates), after
   [collapse_switches] for the arms that are all the same; a
   continuation that is applied once is moved to where it is
   applied and inlined there ([sink_let_conts], then [inline_tail_conts]),
   before an arm of a [Switch] is made to go where its continuation
   goes ([shortcut_switches]), as Simplify makes no shortcut to a
   continuation that it inlines; and before that last, the handler of
   a continuation that is the same as that of a continuation bound
   around it becomes the application of that one
   (FlambdaSameHandlersSpec.share_handlers), which Simplify records
   as a shortcut to it, so that the arms of a [Switch] that do the
   same thing go to one continuation. Then what follows the values,
   into the
   handlers of the recursive continuations too for what is known where
   they are defined ([follow_aliases_in_loops]): among them a
   [Phys_equal] of a simple against itself becomes its answer, as
   Simplify makes it
   (FlambdaSameComparisonSpec.fold_same_comparisons, which makes a
   [Phys_equal] of two different symbols its answer too; it comes before
   what follows the values, which puts the answer in the [Switch] that
   tests it, for the folding of constants that ends the round); a
   test against 0 of the result of a three-way comparison becomes the
   comparison of its two operands, which is what Simplify makes of
   [compare a b < 0]
   (FlambdaRecoverComparisonSpec.recover_known_comparisons; it comes
   before the sharing, so that the comparison it makes is shared with
   one that was computed, and the three-way comparison goes with the
   bindings that are left unused); and a primitive that was computed,
   and a value slot that was projected, is not computed or projected
   again ([share_known_primitives], [share_known_projections]). The
   length of a known immutable array and a load from one are not read
   here: FlambdaValidateReadingSpec.v says why, and has the function
   that reads them. Then
   what removes the bindings and the parameters that are left
   unused; last of all a parameter that is left unused of a
   continuation that a call returns to, the call returning to a
   wrapper that takes what the call gives and passes the rest on,
   which is what Simplify makes
   (FlambdaWrapReturnsSpec.wrap_returns). *)
Definition normalise_expr_before (e : expr) : expr :=
  let e :=
    shortcut_switches
      (share_handlers
         (thread_switches
            (inline_tail_conts
               (sink_let_conts
                  (remove_unused_conts
                     (merge_switches
                        (collapse_switches (fold_constants e))))))))
  in
  let e :=
    apply_identities
      (follow_aliases_in_loops []
         (follow_aliases []
            (share_known_projections
               (share_known_primitives
                  (recover_known_comparisons
                     (fold_same_comparisons
                        (unbox_known_numbers (load_known_blocks e))))))))
  in
  wrap_returns
    (drop_unused_params
       (remove_unused_defined_lets []
          (remove_unused_lets (fold_constants e)))).

(* One round for the body of code in the term before: the round for an
   expression, and the same for the handler of each of its recursive
   continuations, a loop in a loop included. *)
Definition normalise_code_before (body : expr) : expr :=
  rewrite_rec_handlers normalise_expr_before (normalise_expr_before body).

(* One round for the term before, once its calls are inlined and its
   binders have names of their own: the round for the body of code, on
   the term itself; then the static bindings tidied, and that round
   applied to the body of each piece of code. None of these makes a
   name. *)
Definition normalise_before_inlined (e : expr) : expr :=
  rewrite_code normalise_code_before
    (merge_statics (drop_deleted_code (normalise_code_before e))).

(* The twins of the parameters that hold a tagged immediate, for a
   term that a round has left as it was. The parameters of
   continuations that are untagged are bound again at the head of
   their handlers (FlambdaShadowParamsSpec.shadow_params), in the
   term and in the body of each piece of code;
   FlambdaRenameApartSpec.rename_apart_avoiding gives those binders
   names of their own; and FlambdaTwinParamsSpec.twin_params makes
   the twins, in the term and in the body of each piece of code.
   Where a twin was made (the comparison up to bound names,
   FlambdaCompareSpec.expr_eqb, finds a difference) the round for a
   term whose calls are inlined follows; where none was the term is
   given back as it was. Every rewrite refines.

   Why only then. Where the handler of a parameter that holds a
   tagged immediate untags it for a [Switch] whose arms can be
   merged into arithmetic, Simplify, which knows that the parameter
   is the tagging of the number, makes the arithmetic on the
   parameter and has no use for the number. A round here does the
   same in two steps, which may take it two passes: the merge
   (FlambdaMergeSwitchesSpec.merge_switches) tags the scrutinee, and
   the sharing of primitives finds that the tagging of an untagging
   is the variable that was untagged. A twin made before those steps
   takes the place of the untagging, and nothing here knows that the
   tagging of a twin is the parameter. Seen in a unit that is not in
   this tree, at [-O3]: a function on a type of three constant
   constructors that is inlined in another. *)
Definition with_twins (avoid : list continuation) (e : expr) : expr :=
  let shadowed :=
    rename_apart_avoiding avoid (rewrite_code shadow_params (shadow_params e))
  in
  let twinned := rewrite_code twin_params (twin_params shadowed) in
  if expr_eqb [] [] shadowed twinned then e
  else normalise_before_inlined twinned.

(* One round for the term before: the calls of known code inlined, but
   for the code given, in the bodies of code those of the other code of
   its group too (FlambdaInlineSiblingsSpec.inline_sibling_calls); then
   FlambdaRenameApartSpec.rename_apart_avoiding, whose candidate gives
   every binder a name of its own, none of them one of the
   continuations given (the body of a function that is inlined twice
   is copied twice, with the names it binds), and which keeps the
   candidate only where its own comparison finds it the same as the
   term, and the term as it is otherwise; then the round above; and,
   where all that has left the term as it was, up to bound names, the
   twins ([with_twins]). Every rewrite refines. *)
Definition normalise_before_round_avoiding (avoid : list continuation)
    (kept : list code_id) (e : expr) : expr :=
  let e' :=
    normalise_before_inlined
      (rename_apart_avoiding avoid
         (inline_sibling_calls kept (inline_calls kept [] e)))
  in
  if expr_eqb [] [] e e' then with_twins avoid e' else e'.

(* With no continuation to avoid: the round of the comparisons in which
   nothing is lifted, where no check goes by a name that the term does
   not mention. *)
Definition normalise_before_round (kept : list code_id) (e : expr) : expr :=
  normalise_before_round_avoiding [] kept e.

(* One round of the first phase for an expression of the term before:
   [normalise_expr_before] with what Simplify does to the parameters of
   continuations. After an arm of a [Switch] is made to go where its
   continuation goes, every application of a continuation whose handler
   only applies another is, wherever it is, a call that returns to
   it excepted ([forward_applications]; the continuations that this
   leaves unused are removed): among them the applications of a
   continuation that FlambdaSameHandlersSpec.share_handlers has just
   made to apply another, which [shortcut_switches] takes there only
   where they are arms of the [Switch] under its [Let_cont]. Once the
   values are followed, a parameter of a continuation that its handler
   reads as a box, where every application gives something that the
   term shows, is unboxed as Simplify unboxes it: each read becomes a
   parameter, and an application that has nothing to read gives a
   constant in its place
   (FlambdaUnboxVariantsSpec.unbox_variants_knowing, given what is
   known of symbols where the round is applied; [normalise_expr_params]
   is the round with nothing known, in which that is
   FlambdaUnboxVariantsSpec.unbox_variant_params); the boxed
   parameter, where nothing else uses it, is then one of the unused
   ones. The same is then done for a loop: a read of a parameter that
   its handler comes to before any test becomes a parameter, where
   the entry and every turn give something that the term shows
   (FlambdaUnboxLoopsSpec.unbox_loops_knowing, given what is known as
   the other is). Then the parameters that hold a tagged immediate
   and are only untagged or passed on, as a flag that is kept in a
   reference and set in a loop is, are untagged where they are given,
   in all the continuations that pass them at once
   (FlambdaUntagParamsSpec.untag_params). The same is done, last,
   for the parameters that hold a boxed number and are only unboxed
   or passed on, as an accumulator that a loop sets on some turns
   is: an application gives such a parameter another one or the
   number of a box that the term shows
   (FlambdaUnboxTogetherSpec.unbox_together_knowing, given what is
   known as the others are). After the
   bindings that are left unused are removed, a parameter that every
   application gives the same thing is bound to it at the start of the
   handler ([bind_aliased_params]), which leaves it unused as a
   parameter, and the unused ones are removed, those of loops too, and
   with them the parameters of a loop that are only passed round it and
   the [Let]s that only they need ([drop_dead_params]). The next round
   follows the new [Let]s and removes them. *)
Definition normalise_expr_params_knowing (known : facts) (e : expr)
    : expr :=
  let e :=
    shortcut_switches
      (share_handlers
         (thread_switches
            (inline_tail_conts
               (sink_let_conts
                  (remove_unused_conts
                     (merge_switches
                        (collapse_switches (fold_constants e))))))))
  in
  let e := remove_unused_conts (forward_applications e) in
  let e :=
    unbox_together_knowing known
      (untag_params
         (unbox_loops_knowing known
            (unbox_variants_knowing known
               (follow_aliases_in_loops []
                  (follow_aliases []
                     (unbox_known_numbers (load_known_blocks e)))))))
  in
  drop_dead_params
    (bind_aliased_params
       (remove_unused_defined_lets [] (remove_unused_lets (fold_constants e)))).

Definition normalise_expr_params (e : expr) : expr :=
  normalise_expr_params_knowing no_facts e.

(* The same for the body of code: the round for the handler of each of
   its recursive continuations, a loop in a loop included, and then for
   the body itself. The handlers come first: [forward_applications] goes
   into the handler of a loop for the continuations bound outside it,
   and must find those bound inside as [sink_let_conts] and
   [inline_tail_conts] leave them there. The round for a handler is
   given what is known of the symbols in scope where its [Let_cont] is:
   what was known at the start, and the static blocks and boxed numbers
   of the [Let_static]s that the [Let_cont] is under
   (FlambdaUnboxInScopeSpec.rewrite_rec_handlers_in_scope). *)
Definition normalise_code_params_knowing (known : facts) (body : expr)
    : expr :=
  normalise_expr_params_knowing known
    (rewrite_rec_handlers_in_scope normalise_expr_params_knowing known body).

Definition normalise_code_params (body : expr) : expr :=
  normalise_code_params_knowing no_facts body.

(* One round of the first phase for the term before. First, in the
   whole term and in the bodies of its code, a [Switch] whose arms
   give a continuation the elements of a table becomes the load from
   that table, where the table is a static immutable array that is in
   scope (FlambdaMergeTablesSpec.merge_tables, which gives back as it
   is a term that binds no such array); Simplify makes that load, of
   a table that it binds itself. Then the round above on
   the term itself, with nothing known at the start, and on the body of
   each piece of code, with what is known of the symbols in scope where
   the code is bound: the static blocks and boxed numbers of the
   [Let_static]s that the binding is under and of its own group
   (FlambdaUnboxInScopeSpec.rewrite_code_in_scope). So the unboxing of a
   parameter of a continuation knows a constant block of the unit in a
   function and in a loop, as it does under the [Let_static] itself. No
   call is inlined. *)
Definition normalise_params_round (e : expr) : expr :=
  rewrite_code_in_scope normalise_code_params_knowing no_facts
    (normalise_code_params (merge_tables e)).

(* What follows that round in the first phase. The continuations that a
   handler binds on its way to a [Switch] on one of its parameters are
   bound outside the handler, where every application gives that
   parameter a number (FlambdaLiftContsSpec.lift_handler_conts, an
   equivalence), and an application that gives a number to the
   parameter that the handler of its continuation tests goes where the
   arm of that number goes (FlambdaForwardArmSpec.forward_to_arms, a
   refinement). Then, as in the round, an application of a
   continuation whose handler only applies another goes on
   ([forward_applications]), and the continuations that are left
   unused are removed. The forwarding comes at once: the round that
   follows begins by putting a continuation that is applied once
   where it is applied ([sink_let_conts], [inline_tail_conts]), and
   a continuation that was bound outside the handler is applied once,
   by the continuation of a branch, until the application of that one
   is forwarded; the term before would be left with the continuations
   of the branches where the term after has those that were bound
   outside. All this is done in the term, in the handlers of its
   recursive continuations and in the bodies of its code.

   The round above has made the parameters that the handler tests
   where it unboxed a variant, as Simplify does. In the units that
   the step is for, Simplify has left the applications where they
   were (the header of FlambdaForwardArmSpec.v says when it does
   otherwise), and the term after has the first two rewrites, by
   equivalences, in [inline_once_applied_after], which says why.
   They are a step of their own here, and not in the round above, so
   that the round is still what Simplify does to the parameters of
   continuations: the examples of FlambdaUnboxVariantsExamplesSpec.v
   state that rounds of it, or of [normalise_expr_params] that it is
   made of, take a body before to the body that Simplify gave.

   Before them, a parameter of a loop that every turn passes on as it
   is, and that the loop is given one variable for, is taken away,
   and the handler begins with a [Let] of it to that variable
   (FlambdaUnaliasRecParamsSpec.unalias_rec_params, an equivalence,
   which the tidying applies to both terms). The round above binds
   such a parameter where the variable is bound nowhere in the
   [Let_cont] ([bind_aliased_params]), and so leaves one that is
   given the variable of its own name. Every parameter of the loop
   that FlambdaLoopifySpec.loopify_unit puts in the body of code is
   given so: the loop has the parameters of the code and is entered
   with them, and the rounds of the first phase make no names
   apart. *)
Definition arms_expr_before (e : expr) : expr :=
  remove_unused_conts
    (forward_applications
       (forward_to_arms (lift_handler_conts (unalias_rec_params e)))).

Definition arms_code_before (body : expr) : expr :=
  arms_expr_before (rewrite_rec_handlers arms_expr_before body).

Definition first_phase_round (e : expr) : expr :=
  rewrite_code arms_code_before (arms_code_before (normalise_params_round e)).

(* What ends the normalisation of the term before. The exception
   handlers that no call can raise to are made ordinary continuations
   and the trap actions that name them removed, in the term, its
   handlers and its code (FlambdaDemoteSpec.demote_handlers, with no
   name in scope: see the header). Then the rounds of the first phase
   and those of the second, without the inlining of calls, are gone
   through again. They do to what was a [try] what Simplify does on
   its way up, once it has found that the handler can be demoted. The
   continuation of the body, which was applied with the trap action
   that installed the handler, is put where it is applied. A handler
   that nothing raises to is removed. The exception is no longer a
   parameter of a handler that does not read it, and the [Let]s that
   made it go. The continuation by which the body left the [try],
   which only applies another now that its trap action is gone, is a
   shortcut; so is the handler, if it only applies another. *)
Definition demote_and_tidy (rounds : nat) (e : expr) : expr :=
  iterate rounds normalise_before_inlined
    (iterate rounds first_phase_round (demote_handlers [] e)).

(* The term before is normalised in two phases. The first is what
   Simplify does to the continuations of the term as it is: the front
   end makes every mutable variable in scope a parameter of a loop and
   of the continuation after an [if], and Simplify removes those that
   are always given the same thing. The second inlines the calls of
   known code, and the term after has that phase too
   ([normalise_after_round]), also for the calls that Simplify did not
   inline. The work on parameters is not in the second phase: where the
   validation inlines a call that Simplify left, the continuation that
   the call returned to becomes one that the inlined body applies, on
   both sides, and the term after has only equivalences to normalise it
   with, which the removal of a parameter is not. The demotion of
   exception handlers comes after both ([demote_and_tidy]). In a round
   it would come too soon. A round first inlines a continuation that
   is applied once, and then takes an arm of a [Switch] to where its
   continuation goes; so a handler demoted before the raises to it
   have been taken to the arms would be inlined in the continuation
   that raises. Simplify, for which an application with a trap action
   is not one to inline, takes the arm to the handler and leaves the
   handler where it is bound; the rounds do the same while the handler
   has its mark and the raise its trap action.

   Between the rounds of the second phase and the same rounds once
   more come the rewrites that Simplify makes from the types
   (../types/rocq/FlambdaSimplifyPassSpec.simplify_by_types, which
   goes down the whole term with a typing environment: into the
   handlers of continuations, where a handler that is applied once is
   rewritten under the environment of that application, and into the
   bodies of code; it is computed by
   ../types/rocq/FlambdaSimplifyCarriedSpec.simplify_by_types_carried,
   which is that function, of every term, without the test of the
   environment that it makes at every [Let]), under the test of
   FlambdaGuardScopeSpec.guard_scope: of a term whose code is closed,
   what they give is taken only if it keeps scope, and the term is
   left as it was otherwise; of a term whose code is not closed the
   test is not made and what they give is taken. They come once and
   not in every round: after the rounds the term has the shape that
   they read (the calls inlined, a continuation applied once where it
   is applied), and the rounds that follow tidy what they leave, such
   as a handler that has become one application. *)
Definition normalise_before (rounds : nat) (kept : list code_id) (e : expr)
    : expr :=
  demote_and_tidy rounds
    (iterate rounds (normalise_before_round kept)
       (guard_scope FlambdaSimplifyCarriedSpec.simplify_by_types_carried
          (iterate rounds (normalise_before_round kept)
             (iterate rounds first_phase_round e)))).

(* For the term after, what [normalise_expr_before] does first for the
   term before, with rewrites that are equivalences. A continuation
   that is not used is removed. One that is applied once is moved to
   where it is applied, where [sink_let_conts] can move it, and its
   handler is put in place of the application where
   FlambdaInlineTailSpec.inline_tail_conts_kinded allows: the
   application ends the run of [Let]s that is the body of the binding,
   and the handler gives one kind for each of its parameters. The
   validation inlines, in the term after too, the calls that Simplify
   left and that are not in the list of code to leave, and the
   continuation that such a call returned to is then applied once; and
   Simplify leaves some continuations that are applied once, the one
   whose handler binds the symbols of the unit among them. The term
   before can have lost both: its rounds have [sink_let_conts] and
   FlambdaNormaliseSpec.inline_tail_conts.

   Then the continuations that a handler binds on its way to a
   [Switch] on one of its parameters are bound outside the handler,
   where every application gives that parameter a number
   (FlambdaLiftContsSpec.lift_handler_conts); and an application
   that gives a number to the parameter that the handler of its
   continuation tests goes where the arm of that number goes, where
   its arguments are known to have the kinds of the parameters
   (FlambdaForwardArmSpec.forward_to_arms_kinded). Both are
   equivalences. Where the applications give different numbers
   Simplify has a path that does as much where its other tests pass,
   enabled by a flag that no level of optimisation sets (the header
   of FlambdaForwardArmSpec.v says which, and when Simplify decides
   such a [Switch] without it). In the units that the two rewrites
   are for it has left the continuation and its [Switch]; but it
   knows in such a handler what the applications give, and the
   handler that it leaves may differ from that of the term before in
   an arm that no application comes to. The term before
   has the two rewrites in each round of its first phase
   ([first_phase_round]); where every application of the
   continuation then goes to its arm, in both terms, the handler is
   removed from both. An application that goes to an arm gives what
   the arm gives with its own arguments put for the parameters of
   the handler, so a continuation that the arm applies may then be
   given, by every application, one constant or one symbol for a
   parameter (the field of a block that one path built and the
   handler matched). Such a parameter is taken away and bound to
   what it is given at the head of the handler
   (FlambdaBindConstParamsSpec.bind_const_params, an equivalence),
   and the rewrites of the round fold what reads it; the term before
   has the same in the next round of its first phase, from
   FlambdaParamAliasesSpec.bind_aliased_params, which refines.

   Then an application of a continuation whose handler is one
   [Apply_cont] becomes that [Apply_cont], with the arguments put for
   the parameters, where FlambdaForwardKindedSpec.forward_kinded
   allows: among its conditions, each argument is known to have the
   kind that the handler of the continuation gives the parameter it
   is given to (a variable by the [Let] or the handler that binds it
   in the term, a constant by itself, a symbol being of kind
   [Value]), the handler gives as many kinds as it has parameters,
   the [Apply_cont] of the handler has no trap action (the
   application keeps its own, if it has one), and nothing that the
   [Apply_cont] of the handler names is bound again in the body of
   the [Let_cont].
   It is an equivalence. A continuation that this leaves unused is
   removed. Simplify leaves such an application in three cases. The
   handler was a [Switch] on a parameter that it removed, every
   application giving it the same constant: the handler it has is
   that [Switch] under the [Let] of the parameter, and the fold of
   the round before has made it one [Apply_cont]. Or every
   application gave a parameter the same variable, bound in the body
   of the [Let_cont]: Simplify passes that variable in its place and
   binds the old name at the head of the handler, and the round
   before has followed that [Let] and removed it (a loop in a
   function leaves so for the return continuation with what it has
   computed). Or the handler ended a region before it passed its
   parameter on, which the model does not have (the continuation
   that follows a [try] whose result is matched, or is returned: the
   body of the [try] then applies it with the trap action that pops
   the handler). In all three the
   term before has the same handler, and
   FlambdaForwardSpec.forward_applications has forwarded its
   applications.

   Then an arm of a [Switch] that applies a continuation of no
   parameters whose handler is a [Switch] on the same scrutinee goes
   where the arm of the handler for the same number goes, where the
   checks of FlambdaNormaliseSpec.thread_switches permit (an
   equivalence), as a round of the term before has it. The term
   after has such a pair in a generated unit: the scrutinee of the
   two is what the arms of a third [Switch], which Simplify merged,
   gave two parameters. *)
Definition inline_once_applied_after (e : expr) : expr :=
  remove_unused_conts
    (thread_switches
       (forward_kinded
          (FlambdaBindConstParamsSpec.bind_const_params
             (forward_to_arms_kinded
                (lift_handler_conts
                   (inline_tail_conts_kinded
                      (sink_let_conts (remove_unused_conts e)))))))).

(* The same for the term after. Every rewrite is an equivalence.
   Before the aliases are followed, a [Let] of a variable to the
   variable that the [Let] around it has just bound, and that is not
   free in what follows, is made one [Let] with it
   (FlambdaAliasLetsSpec.merge_alias_lets): the copy of a body that is
   inlined ends with such a [Let] of the result of the call, and the
   aliases cannot be followed past a second copy of the same body,
   which binds the same names, as nothing is renamed apart in the term
   after. The
   unused [Let]s that are sure to have a value are removed in the
   handlers of recursive continuations too, with what is in scope there
   ([remove_unused_defined_lets_in_loops]): Simplify leaves such a [Let]
   where it has removed a parameter that every application gave the same
   thing. Before that, the [Let] of the projection of a function slot
   that nothing uses is removed
   (FlambdaUnusedProjectionsSpec.remove_unused_projections): the [Let]
   of a closure of a set that is not the first is one in the model
   (FlambdaModel.let_closures), and the term before loses it by a
   refinement. Before the constants are folded, what a known block
   holds is read where the read is sure to have a value
   (FlambdaReadsAfterSpec.load_known_blocks_surely), and after the
   unused projections an allocation of constants that nothing uses is
   removed (FlambdaReadsAfterSpec.remove_constant_allocations):
   Simplify leaves the load of a field of a block that a call it did
   not inline is given, and the block with it; where the validation
   inlines that call in the two terms, the term before loses both by
   refinements (FlambdaBlockLoadSpec.load_known_blocks,
   FlambdaNormaliseSpec.remove_unused_lets). Between that read and
   the folding of constants, the length of a mutable array that a
   [Let] in scope has made becomes its constant
   (FlambdaMadeLengthsSpec.fold_made_lengths). The rounds of the term
   before do that where they share primitives
   (FlambdaCseSpec.share_known_primitives), which these rounds do not
   do: where the validation inlines a call in the two terms, the term
   before loses the length of an array made at the call, the
   comparisons on it and the arms that those decide, and the term
   after must lose them too. Nothing else of that sharing is applied
   here. Last in a round come two
   passes that go into the
   bodies of code with the parameters of the code in scope, at the
   kinds of its arity. The applications that
   FlambdaForwardKindedSpec.forward_kinded forwards are forwarded
   there ([forward_kinded_in_code]): in [inline_once_applied_after]
   the pass starts in the body of code with nothing known, and leaves
   an application that gives a continuation a parameter of the code
   (a function that returns its argument where a [try] has failed).
   The same is done for the applications that
   FlambdaForwardArmSpec.forward_to_arms_kinded makes go to an arm
   ([forward_to_arms_in_code]).
   Then the unused [Let]s that are sure to have a value are removed
   (FlambdaCodeLetsSpec.remove_unused_defined_lets_in_code): where
   Simplify makes a loop of a function, it leaves such a [Let] of each
   parameter of the function that every self tail call passes on as it
   is. *)
Definition normalise_expr_after (e : expr) : expr :=
  remove_unused_defined_lets_in_loops []
    (remove_constant_allocations
       (remove_unused_projections
          (fold_constants
             (fold_made_lengths (load_known_blocks_surely
                (follow_aliases_in_loops []
                   (follow_aliases []
                      (merge_alias_lets
                         (inline_once_applied_after e))))))))).

Definition normalise_code_after (body : expr) : expr :=
  rewrite_rec_handlers normalise_expr_after (normalise_expr_after body).

(* A round of the term after. The calls of known code are inlined
   where that is an equivalence
   (FlambdaNormaliseSpec.inline_known_calls); then, in the body of
   code, so are those of the other code of its group
   (FlambdaKnownSiblingsSpec.inline_known_sibling_calls), which a
   round of the term before inlines too
   (FlambdaInlineSiblingsSpec.inline_sibling_calls): Simplify leaves
   the call of a function in the code of another function of its
   group, where it inlines the function at a call that is not in the
   code of the group. *)
Definition normalise_after_round (kept : list code_id) (e : expr) : expr :=
  remove_unused_defined_lets_in_code
    (forward_to_arms_in_code
       (forward_kinded_in_code
          (rewrite_code normalise_code_after
             (merge_statics
                (drop_deleted_code
                   (rewrite_rec_handlers normalise_expr_after
                      (remove_unused_defined_lets_in_loops []
                         (remove_constant_allocations
                            (remove_unused_projections
                               (fold_constants
                                  (fold_made_lengths (load_known_blocks_surely
                                     (inline_known_sibling_calls kept
                                        (inline_known_calls kept [] []
                                           (follow_aliases_in_loops []
                                              (follow_aliases []
                                                 (merge_alias_lets
                                                    (inline_once_applied_after
                                                       e)))))))))))))))))).

Definition normalise_after (rounds : nat) (kept : list code_id) (e : expr)
    : expr :=
  iterate rounds (normalise_after_round kept) e.

(* The static bindings of the names given removed, if nothing refers to
   them, and the groups of bindings tidied again. *)
Definition without_statics (D : dropped) (e : expr) : expr :=
  match drop_statics D e with
  | Some e' => merge_statics (drop_deleted_code e')
  | None => e
  end.

(* For the term after: [without_statics], where the checks of
   FlambdaUnusedHeadStaticsSpec.head_statics_removable hold. The
   bindings removed are then among those of the groups at the head of
   the term, and the term, run on its own, refines the result
   ([drop_head_statics_program_refines_statement] of that file). *)
Definition without_head_statics (D : dropped) (e : expr) : expr :=
  if head_statics_removable D e then without_statics D e else e.

(* The [Let]s given made static bindings of the symbols given: those
   to bind at the head of the unit, if the conditions of
   FlambdaLiftFromArmsSpec.lift_to_head_unit hold; then those in the
   bodies of code, if the conditions of
   FlambdaLiftFromCodeSpec.lift_from_code_unit hold of what that gives;
   then those outside code that are bound where they stand, if the
   conditions of FlambdaLiftSpec.lift_unit hold of what that gives;
   then those that are made aliases of symbols that a group around
   them binds, with the groups that the guess puts, if the conditions
   of FlambdaLiftAroundSpec.lift_around_unit hold of what that gives.
   Where the conditions of one do not hold, it is not made. The paths
   of each are paths in the term that it is given: the first puts the
   unit under a group of static bindings, if it lifts anything. After
   those, the tables given are bound in a group at the head of the
   unit, where the checks of
   FlambdaUnusedHeadStaticsSpec.add_head_tables pass: Simplify binds
   such a table where it makes a load of a [Switch] whose arms give
   constants, and a round that makes the same load of the term before
   needs the table in scope. As it comes after them, no path of the
   liftings before it is a path in a term that has the group. Last of
   all, the copies of code given are made, each where the check of
   FlambdaCopyCodeSpec.copy_static_code passes: a list of code ids,
   each paired with a new one, and the symbols of the static sets of
   closures that get the new ones. As it comes last, those symbols
   may be ones that the liftings before it have just bound; it is
   given no path. With no copy given, the result is what it was
   without this step (FlambdaCopyCodeSpec.with_copies_none_statement). *)
Definition with_lifting (L : lifting) (e : expr) : expr :=
  let e :=
    match lift_to_head_unit (lifting_unit L) (lifted_to_head L) e with
    | Some e' => e'
    | None => e
    end
  in
  let e :=
    match lift_from_code_unit (lifting_unit L) (lifted_out_of_code L) e with
    | Some e' => e'
    | None => e
    end
  in
  let e :=
    match lift_unit L e with
    | Some e' => e'
    | None => e
    end
  in
  let e :=
    match
      lift_around_unit (lifting_unit L) (lifted_around L) (groups_put L) e
    with
    | Some e' => e'
    | None => e
    end
  in
  with_copies (code_copied L) (add_head_tables (tables_put L) e).

(* What the unit binds its symbols to used wherever they are, if it
   binds each of them to one constant. *)
Definition with_known (e : expr) : expr :=
  match fold_known_unit e with
  | Some e' => e'
  | None => e
  end.

(* The normalisation of a term in which something was lifted: the first
   phase of that of the term before, then its second phase with each
   round followed by the use of what is known of the static closures in
   sight, of what the unit binds its symbols to, and of the value slots
   that every set of closures of the unit gives one symbol or
   constant, and by the sharing of the reads of the places that one
   group of static bindings gave one variable
   (FlambdaSharedReadsSpec.share_static_reads).

   Between those rounds and the same rounds once more come the
   rewrites that Simplify makes from the types, under the same test
   as in [normalise_before], and once for the same reason. In the
   case observed, the unit keeps a function and the comparison that
   succeeds for it is the one of the lifted terms; the lifted term
   before is normalised here from the term before as it was read and
   not from what [normalise_before] gives, so without the rewrites
   here what they do to such a unit is lost. The case: a [Switch] on
   a variable, in the code of a function that the unit keeps, in a
   handler that only one arm of a [Switch] on the same variable
   applies. Not every unit that keeps a function is validated by
   that comparison. *)
Definition normalise_lifted_round (avoid : list continuation)
    (kept : list code_id) (e : expr) : expr :=
  share_static_reads
    (read_known_slots
       (with_known
          (use_known_closures
             (normalise_before_round_avoiding avoid kept e)))).

Definition normalise_lifted (rounds : nat) (avoid : list continuation)
    (kept : list code_id) (e : expr) : expr :=
  iterate rounds (normalise_lifted_round avoid kept)
    (guard_scope FlambdaSimplifyCarriedSpec.simplify_by_types_carried
       (iterate rounds (normalise_lifted_round avoid kept)
          (iterate rounds first_phase_round e))).

(* The calls of the closures that a [Let] makes made direct, in the
   term and in the bodies of its code
   (FlambdaClosureCallsSpec.call_made_closures, which does not go into
   code). *)
Definition with_made_closures_called (e : expr) : expr :=
  rewrite_code call_made_closures (call_made_closures e).

(* A round of [normalise_lifted] and then, where [calling] is set,
   those calls made direct in the term that the round gives. Simplify
   makes such a call direct where it comes to it, from the type of the
   closure, and may then inline it; the case observed is a function
   that makes two closures and gives them back in a block, inlined in
   a round at two calls: the call of a closure read from the block is
   then the call of a variable that a [Let] binds. A call that this
   makes direct is one that the next round may inline. The round of
   [normalise_lifted] is named here and not written again, so what
   that round does this one does. *)
Definition normalise_lifted_round_calling (calling : bool)
    (avoid : list continuation) (kept : list code_id) (e : expr) : expr :=
  let round := normalise_lifted_round avoid kept e in
  if calling then with_made_closures_called round else round.

(* [normalise_lifted] with those rounds. With [calling] unset it is
   [normalise_lifted], by computation
   ([normalise_lifted_calling_unset_statement]). *)
Definition normalise_lifted_calling (calling : bool) (rounds : nat)
    (avoid : list continuation) (kept : list code_id) (e : expr) : expr :=
  iterate rounds (normalise_lifted_round_calling calling avoid kept)
    (guard_scope FlambdaSimplifyCarriedSpec.simplify_by_types_carried
       (iterate rounds (normalise_lifted_round_calling calling avoid kept)
          (iterate rounds first_phase_round e))).

Definition normalise_lifted_calling_unset_statement : Prop :=
  ∀ (rounds : nat) (avoid : list continuation) (kept : list code_id)
    (e : expr),
    normalise_lifted_calling false rounds avoid kept e
    = normalise_lifted rounds avoid kept e.

(* A round of [normalise_lifted_calling] and then, where [own] is
   set, what the closures of the code of the unit are known to be
   used in the bodies of that code, in the term that the round gives
   (FlambdaClosureOfCodeSpec.know_own_closures): a call of
   [my_closure], or of another closure of its set projected from it,
   is direct, and a value slot of [my_closure] that every set with its
   code gives one symbol or constant is that. A call of a value slot
   of [my_closure] that every set with its code gives a closure in
   sight of one code is direct too (FlambdaSlotsOfCodeSpec.v).
   Simplify does all three from the type that it gives [my_closure]
   in the code of a set that it simplifies. The sets are those that
   the term of the round makes, so a set that a lifting has made
   static, or a round has changed, is counted as it then is. A call
   that this makes direct is one that the next round may inline. The
   round of [normalise_lifted_calling] is named here and not written
   again. *)
Definition normalise_lifted_round_own (calling own : bool)
    (avoid : list continuation) (kept : list code_id) (e : expr) : expr :=
  let round := normalise_lifted_round_calling calling avoid kept e in
  if own then know_own_closures round else round.

(* [normalise_lifted_calling] with those rounds. With [own] unset it
   is [normalise_lifted_calling], by computation
   ([normalise_lifted_own_unset_statement]); so with both unset it is
   [normalise_lifted]. *)
Definition normalise_lifted_own (calling own : bool) (rounds : nat)
    (avoid : list continuation) (kept : list code_id) (e : expr) : expr :=
  iterate rounds (normalise_lifted_round_own calling own avoid kept)
    (guard_scope FlambdaSimplifyCarriedSpec.simplify_by_types_carried
       (iterate rounds (normalise_lifted_round_own calling own avoid kept)
          (iterate rounds first_phase_round e))).

Definition normalise_lifted_own_unset_statement : Prop :=
  ∀ (calling : bool) (rounds : nat) (avoid : list continuation)
    (kept : list code_id) (e : expr),
    normalise_lifted_own calling false rounds avoid kept e
    = normalise_lifted_calling calling rounds avoid kept e.

(* The return and the exception continuation of the unit of each of
   the liftings given: the names by which a lifting checks the unit
   (FlambdaLiftSpec.lift_unit), which the term need not mention. *)
Definition unit_continuations (Ls : list lifting) : list continuation :=
  flat_map
    (fun L => [unit_return (lifting_unit L); unit_exn (lifting_unit L)]) Ls.

(* What ends the normalisation of a term in which something was
   lifted. The exception handlers are demoted and the rounds gone
   through again, as at the end of [normalise_before]
   ([demote_and_tidy]). A handler that is demoted is a continuation
   like another, and the rounds may then put a symbol that a lifting
   bound where the handler read its parameter. The case observed: a
   function raises an exception that it makes, a block that the
   lifting binds to a symbol beside the code, and the handler of a
   [try] around the raise loads the first field of what it is given
   and tests it against the constructor. Simplify, which knows what
   the symbol is bound to when it comes to the handler, gives the
   load its value and so resolves the test and the [Switch] on it;
   the rounds of [normalise_lifted], which use what is known of the
   symbols, are over before the demotion. So what
   [normalise_lifted_round] uses after its round is used here, in
   rounds of its own, each followed by [demote_and_tidy] again: what
   is known of the static closures in sight, what the unit binds its
   symbols to, the value slots that every set of closures of the
   unit gives one symbol or constant, and the places that one group
   of static bindings gave one variable. No call is inlined and
   nothing is renamed apart, so these rounds make no name; a call
   that they make direct stays a call. *)
Definition demote_and_tidy_known_round (rounds : nat) (e : expr) : expr :=
  demote_and_tidy rounds
    (share_static_reads
       (read_known_slots (with_known (use_known_closures e)))).

Definition demote_and_tidy_known (rounds : nat) (e : expr) : expr :=
  iterate rounds (demote_and_tidy_known_round rounds)
    (demote_and_tidy rounds e).

(* The term with each of the liftings given applied in turn, each
   followed by the normalisation of a term in which something was
   lifted, with the calls of the closures that a [Let] makes made
   direct in its rounds where the lifting says so
   ([normalise_lifted_calling]), and with what [my_closure] is known
   to be from its code used in them where the lifting says so
   ([normalise_lifted_own]). A lifting after the first names
   [Let]s of the term that those before it give. The continuations to
   avoid are the same at every step: [validate_unit] gives those of
   the units of all the liftings, so that no round, before or after
   any of them, makes one. After the last, the value slots that are
   known are read once more, with every symbol that the unit binds
   taken to be in scope
   (FlambdaKnownSlotsSpec.read_known_slots_anywhere): Simplify reads a
   slot also in code that the text of the unit binds above the symbol
   that the slot holds, and no lifting comes after this to ask row 4
   of the term. What it gives need not pass row 4; nothing that
   [validate_unit] does after it relies on row 4 without testing it.
   Then the exception handlers are demoted, as at the end of
   [normalise_before], and what is known of the symbols is used again
   ([demote_and_tidy_known]); the rounds that follow the demotion
   inline no call and rename nothing apart, so they make no name. *)
Fixpoint lifted_in_turn (rounds : nat) (avoid : list continuation)
    (Ls : list lifting) (kept : list code_id) (e : expr) : expr :=
  match Ls with
  | [] => demote_and_tidy_known rounds (read_known_slots_anywhere e)
  | L :: Ls =>
    lifted_in_turn rounds avoid Ls kept
      (normalise_lifted_own (made_closures_called L) (own_closures_known L)
         rounds avoid kept (with_lifting L e))
  end.

(* The value slots given removed from the static sets of closures, if
   nothing projects them; and then the [Let]s that nothing uses, in the
   term and in the bodies of its code
   (FlambdaNormaliseSpec.remove_unused_lets). The slots are removed
   after the last round, so a [Let] whose only use was a slot that is
   removed here is one that no round has looked at since: the closure
   of a function that was held in a slot of another closure and
   called through it, once that call is inlined. Simplify removes the
   slot, the closure and its code; with the [Let] gone, the code is a
   static binding that nothing names, for [without_statics] to
   remove. *)
Definition without_slots (slots : list value_slot) (e : expr) : expr :=
  match drop_unused_slots slots e with
  | Some e' => rewrite_code remove_unused_lets (remove_unused_lets e')
  | None => e
  end.

(* The unit with all its code bound first, if the conditions of
   FlambdaHoistCodeSpec.hoist_code hold, and its groups of static
   bindings tidied again. *)
Definition with_code_first (e : expr) : expr :=
  match hoist_code e with
  | Some e' => merge_statics (drop_deleted_code e')
  | None => e
  end.

(* The unit with its static bindings that need nothing from around
   them bound first, the code and the symbols whose constants mention
   no variable, if the conditions of
   FlambdaHoistStaticsSpec.hoist_statics hold for the unit of the first
   of the liftings given; and its groups of static bindings tidied
   again. The term as it is if no lifting is given. *)
Definition with_statics_first (Ls : list lifting) (e : expr) : expr :=
  match Ls with
  | [] => e
  | L :: _ =>
    match hoist_statics (lifting_unit L) e with
    | Some e' => merge_statics (drop_deleted_code e')
    | None => e
    end
  end.

(* What is done once, when the rounds are over, to each of the terms
   that are then compared; the first comparison, of the two terms as
   they are given, comes before it. A non-recursive [Let_cont] goes
   down past the bindings under it, and into the handler under it
   that alone names its continuation, as far as
   FlambdaSinkUsesSpec.sink_conts_to_uses can move it: in the term and
   in the handlers of its recursive continuations ([place_conts]), and
   the same in the body of each piece of code. Not in a round: the
   rewrites of a round that look at where a continuation is bound
   follow Simplify on the term as the rounds leave it. *)
Definition place_conts (e : expr) : expr :=
  rewrite_rec_handlers sink_conts_to_uses (sink_conts_to_uses e).

Definition placed (e : expr) : expr :=
  rewrite_code place_conts (place_conts e).

(* What is done to each of the terms that are compared once its
   continuations are placed, in rounds. One round for an expression of
   the term before: an argument that is the number of the arm that it
   is in, of a [Switch] on a variable, is made that variable where a
   parameter can then be taken away
   (FlambdaGiveScrutineeSpec.give_scrutinees); a parameter of a
   continuation that every
   application gives one variable, which is at the handler what it is
   at the applications, is taken away where the checks of
   FlambdaUnaliasParamsSpec.unalias_params pass, and the handler
   begins with a [Let] of it to that variable; so is a parameter of
   a loop that every turn passes on and that the loop is given one
   variable for (FlambdaUnaliasRecParamsSpec.unalias_rec_params); the
   variable that a [Switch] tests is made the number of the arm, in
   the arm and in the handler that only that arm applies, at the
   reads that FlambdaArmNumbersSpec.v lists, and the variable that
   it is the untagging of is made the tagged number at those that
   the file lists for it (FlambdaArmNumbersSpec.numbers_in_arms). A
   primitive may then be on constants and a [Switch] on a constant,
   as they are for Simplify, which simplifies the handler knowing the
   number: they are folded ([fold_constants]), an application of a
   continuation whose handler is then one [Apply_cont] is made that
   [Apply_cont], with its arguments put for the parameters
   (FlambdaForwardSpec.forward_applications), a [Switch] whose arms
   are then all the same is that arm ([collapse_switches]), and a
   [Switch] whose arms then apply one continuation to arguments that
   are computed from their numbers is made the application
   (FlambdaMergeSwitchesSpec.merge_switches); a continuation that
   this leaves unused is removed, and one that it leaves applied
   once is moved to its application and its handler put there
   ([remove_unused_conts], [sink_let_conts], [inline_tail_conts]): a
   round of the normalisation does as much after its own fold, and
   no round of it follows the placing. Then a primitive that was
   computed is not computed again; aliases are followed, into the
   handlers of recursive continuations too for what is known where
   they are defined; the [Let]s left unused are removed. *)
Definition tidy_expr_before (e : expr) : expr :=
  remove_unused_defined_lets []
    (remove_unused_lets
       (follow_aliases_in_loops []
          (follow_aliases []
             (share_known_primitives
                (inline_tail_conts
                   (sink_let_conts
                      (remove_unused_conts
                         (merge_switches
                            (collapse_switches
                               (forward_applications
                                  (fold_constants
                                     (numbers_in_arms
                                        (unalias_rec_params
                                           (unalias_params
                                              (give_scrutinees
                                                 e))))))))))))))).

(* The same for the body of code: that round, and the same for the
   handler of each of its recursive continuations. *)
Definition tidy_code_before (body : expr) : expr :=
  rewrite_rec_handlers tidy_expr_before (tidy_expr_before body).

(* One round for the term before: that on the term itself, and on the
   body of each piece of code. *)
Definition tidy_before_round (e : expr) : expr :=
  rewrite_code tidy_code_before (tidy_code_before e).

(* The term before as it is compared, when the rounds of the
   normalisations are over: its continuations placed, then the
   rounds above, and then its groups of static bindings bound before
   the [Let_cont]s that they are at the head of
   (FlambdaGroupsUpSpec.groups_up). *)
Definition tidied_before (rounds : nat) (e : expr) : expr :=
  groups_up (iterate rounds tidy_before_round (placed e)).

(* The same for the term after, with rewrites that are equivalences
   only: the arms of a [Switch] are not merged, nor a [Switch] whose
   arms are all the same made that arm; the handler of a
   continuation applied once is put at its application where
   FlambdaInlineTailSpec.inline_tail_conts_kinded allows; and of the
   [Let]s left unused, those that are sure to have a value are
   removed, as in the normalisation of the term after.

   After the fold, an application of a continuation whose handler is
   then one [Apply_cont] is made that [Apply_cont] where
   FlambdaForwardKindedSpec.forward_kinded allows, as the term before
   has it by [forward_applications] in [tidy_expr_before]: without
   it the term before may have an application gone on where the term
   after still has the continuation. Then the continuations that a
   handler binds on its way to a [Switch] on one of its parameters
   are bound outside it, and an application that gives that
   parameter a number goes to the arm of the number, where its
   arguments are known to have the kinds of the parameters
   (FlambdaLiftContsSpec.lift_handler_conts,
   FlambdaForwardArmSpec.forward_to_arms_kinded), as in the rounds
   of the term after ([inline_once_applied_after]). They come again
   here because the rounds may leave a continuation that the first
   of the two refuses and that the rewrites above then make one that
   it accepts, when no round follows. It refuses where an
   application gives a variable for a parameter that its arm does
   not read (FlambdaLiftContsSpec.read_where_given); and where every
   application gives that parameter one variable, [unalias_params]
   above takes the parameter away. The term before has its
   applications go to their arms in the rounds of its first phase
   ([first_phase_round]). *)
Definition tidy_expr_after (e : expr) : expr :=
  remove_unused_defined_lets_in_loops []
    (follow_aliases_in_loops []
       (follow_aliases []
          (share_known_primitives
             (inline_tail_conts_kinded
                (sink_let_conts
                   (remove_unused_conts
                      (forward_to_arms_kinded
                         (lift_handler_conts
                            (forward_kinded
                               (fold_constants
                                  (numbers_in_arms
                                     (unalias_rec_params
                                        (unalias_params
                                           (give_scrutinees e)))))))))))))).

Definition tidy_code_after (body : expr) : expr :=
  rewrite_rec_handlers tidy_expr_after (tidy_expr_after body).

Definition tidy_after_round (e : expr) : expr :=
  remove_unused_defined_lets_in_code
    (rewrite_code tidy_code_after (tidy_code_after e)).

Definition tidied_after (rounds : nat) (e : expr) : expr :=
  groups_up (iterate rounds tidy_after_round (placed e)).

(* The calls of the term before, when its rounds are over. A call
   that returns to a continuation whose handler passes its
   parameters on as they are is made to return to the continuation
   that the handler applies
   (FlambdaForwardSpec.forward_applications_with, with the calls),
   and the continuations that this leaves unused are removed: in a
   term and in the handlers of its recursive continuations
   ([take_returns_on]), and the same in the body of each piece of
   code. *)
Definition take_returns_on (e : expr) : expr :=
  rewrite_rec_handlers remove_unused_conts
    (remove_unused_conts (forward_applications_with true e)).

Definition returns_taken_on (e : expr) : expr :=
  rewrite_code take_returns_on (take_returns_on e).

(* Whether a term has a call that would be taken on: the forwarding
   with the calls, in the term and in the bodies of its code, gives
   another term than the forwarding without them. *)
Definition forwarded_in_unit (calls : bool) (e : expr) : expr :=
  rewrite_code (forward_applications_with calls)
    (forward_applications_with calls e).

Definition keeps_a_return (e : expr) : bool :=
  negb
    (expr_eqb [] [] (forwarded_in_unit true e) (forwarded_in_unit false e)).

(* The term before with its calls taken on, or as it is. [taken] is
   for [validate_unit] to choose: it says no where the term after, as
   it is compared, has a call that would be taken on, which this
   rewrite cannot in general take on by an equivalence. *)
Definition with_returns (taken : bool) (e : expr) : expr :=
  if taken then returns_taken_on e else e.

(* ======================================================================== *)
(* The check                                                                *)
(* ======================================================================== *)

(* How a pair of terms was found to be correct, if it was.
   [Same_once_unused_statics_removed] is of the term after as it is
   once normalised, placed and tidied: the bindings are removed from
   the term before only. [Same_once_head_statics_removed] is of the
   term after with unused static bindings removed from its head too
   ([without_head_statics]), and so is [Same_once_lifted] where there
   are such bindings. *)
Inductive validation : Type :=
  | Same_up_to_bound_names
  | Same_once_normalised
  | Same_once_unused_statics_removed
  | Same_once_head_statics_removed
  | Same_once_lifted
  | Not_validated.

Definition validate_unit (rounds : nat) (unused : dropped)
    (slots : list value_slot) (G : loopified) (Ls : list lifting)
    (kept : list code_id) (before after : expr) : validation :=
  if expr_eqb [] [] before after then Same_up_to_bound_names
  else
    let before :=
      loopify_unit G (cut_returns [] (demote_stub_calls before))
    in
    let after := tidied_after rounds (normalise_after rounds kept after) in
    let taken := negb (keeps_a_return after) in
    let normalised :=
      tidied_before rounds
        (with_returns taken (normalise_before rounds kept before))
    in
    if expr_eqb [] [] (order_statics_as after normalised) after
    then Same_once_normalised
    else if
      expr_eqb [] []
        (order_statics_as after (without_statics unused normalised)) after
    then Same_once_unused_statics_removed
    else
    let after := without_head_statics unused after in
    if
      expr_eqb [] []
        (order_statics_as after (without_statics unused normalised)) after
    then Same_once_head_statics_removed
    else if
      expr_eqb [] []
        (order_statics_as (float_code (with_code_first after))
           (float_code
              (without_statics unused
                 (without_slots slots
                    (tidied_before rounds
                       (with_returns taken
                          (lifted_in_turn rounds (unit_continuations Ls) Ls
                             kept before)))))))
        (float_code (with_code_first after))
    then Same_once_lifted
    else if
      expr_eqb [] []
        (order_statics_as (with_statics_first Ls after)
           (with_statics_first Ls
              (without_statics unused
                 (without_slots slots
                    (tidied_before rounds
                       (with_returns taken
                          (lifted_in_turn rounds (unit_continuations Ls) Ls
                             kept before)))))))
        (with_statics_first Ls after)
    then Same_once_lifted
    else Not_validated.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* A unit that is validated refines the unit it came from, run on its own
   from empty environments. *)
Definition validate_unit_sound_statement : Prop :=
  ∀ (rounds : nat) (unused : dropped) (slots : list value_slot)
    (G : loopified) (Ls : list lifting) (kept : list code_id)
    (before after : expr),
    validate_unit rounds unused slots G Ls kept before after
    <> Not_validated ->
    program_refines before after.

(* It does so under all environments if no static binding was removed and
   nothing was lifted. *)
Definition validate_unit_sound_everywhere_statement : Prop :=
  ∀ (rounds : nat) (unused : dropped) (slots : list value_slot)
    (G : loopified) (Ls : list lifting) (kept : list code_id)
    (before after : expr),
    validate_unit rounds unused slots G Ls kept before after
    = Same_up_to_bound_names
    ∨ validate_unit rounds unused slots G Ls kept before after
      = Same_once_normalised ->
    expr_refines before after.

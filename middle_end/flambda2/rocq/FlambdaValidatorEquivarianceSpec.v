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


(* The rewrites and the checks that the validator is made of commute
   with renaming.

   This file is to be reviewed: [rewrite_equivariant], the action of a
   renaming on what a pass carries (the definitions named [rename_...]),
   and the statements.

   FlambdaEquivarianceSpec.v says that execution commutes with the
   renamings of FlambdaNominalSpec.v, and that two of the rewrites that
   the validation applies do ([fold_constants], [inline_calls]). Here are
   the statements for the other rewrites and for the checks of the files
   listed below, in the forms that a function on terms owes (NAMES.md,
   section 7):
   - a rewrite that takes a term and nothing else gives, on the renamed
     term, the renamed result ([rewrite_equivariant]);
   - a rewrite that takes names too, or something that holds names (the
     variables in scope, the code known, what a pass knows of the
     variables of the term), gives the renamed result when those are
     renamed with the term;
   - a check answers the same on the renamed term, the names it is given
     being renamed with it.
   Symbols, code ids and slots are not renamed. A map keyed by them
   keeps its keys; what it holds is renamed where it holds variables or
   continuations (the code known, under its code id; what is known of a
   symbol that is a block), and is as it was where it holds none (the
   names to remove, the code id of a symbol known to be a closure).

   The normalisation that the validation applies to the term after
   (FlambdaValidateSpec.normalise_after), the removal of static
   bindings ([without_statics]) and what a round of the term before
   does once the calls are inlined and the binders have names of their
   own ([normalise_before_inlined]) are made of these rewrites and
   commute with renaming too. The comparison of the two results
   answers the same when both are renamed
   (FlambdaCompareAlphaSpec.expr_eqb_equivariant_statement).

   The verdict does not depend on the names of the unit, where the
   code of the term before is closed. This file stated it of every
   pair of terms, while no rewrite of the validation made a name: the
   normalisation of the term before
   (FlambdaValidateSpec.normalise_before) commuted with renaming, and
   the check as a whole, FlambdaValidateSpec.validate_unit, answered
   the same on a pair of terms and on the pair renamed. A round of the
   term before now gives every binder a name of its own
   (FlambdaRenameApartSpec.rename_apart), and that rewrite makes
   names: it commutes with a renaming only up to the names that it
   makes. So the first statement is no longer here, and the second is
   of fewer terms. The last section says that the normalisation of the
   term before commutes with renaming up to the names that it makes,
   and that the check answers the same on a pair of terms and on the
   pair renamed where the code of the term before is closed, as that
   of a well-formed unit is ([validate_unit_equivariant_statement]).
   Nothing is said of a term before with open code.

   Each rewrite and each check of the sections before the one headed
   "Rewrites that make names" depends on the names of the term by
   their equality only: none goes by the order or the number of a
   name, and none makes a name. The one place where names are made
   from others is the renaming that inlining builds from the four
   continuations of a call and of the code called
   (FlambdaNormaliseSpec.inline_renaming); it is made of swaps of those
   four, and commutes with a renaming of them.

   Every statement of those sections is an equation and holds of every
   term: nothing is asked of the term, its code, its binders or its
   scopes. Four statements have a premise, on an argument that is a
   function: those of [rewrite_code], of [rewrite_rec_handlers] and of
   [iterate] ask that the rewrite they are given commutes with
   renaming, and that of [static_names] asks that the way it is given
   of reading the names of a binding does not look at variables and
   continuations. The statements of the two sections after them,
   "Rewrites that make names" and the last, are not all equations, and
   are of terms whose code is closed (FlambdaBindersSpec.code_closed),
   as that of a well-formed unit is.

   Kinds, arities and the metadata of handlers and of code have no
   names in them: a renaming keeps them, and a check that reads them
   answers the same.

   The rewrites and checks of FlambdaNormaliseSpec.v:
   - the checks [expr_in_scope], [inlinable_b], [tupled_inlinable_b]
     and [var_free_b], and the test of whether a call agrees with the
     code called in its arities ([call_agrees_b]);
   - the rewrites [remove_unused_lets], [inline_tail_conts] (with
     [inline_cont_at_tail], which it is made of), [drop_deleted_code],
     [remove_unused_total_lets], [remove_unused_defined_lets],
     [inline_known_calls], [merge_statics], [collapse_switches] and
     [thread_switches];
   - [rewrite_code], which takes a rewrite into the bodies of code: it
     is equivariant when the rewrite it is given is.

   Those of FlambdaUnusedContsSpec.v: the check [cont_free_b] and the
   rewrite [remove_unused_conts].

   Those of FlambdaUnusedStaticsSpec.v, where the names given are
   symbols and code ids: the checks [expr_avoids] and
   [binds_only_kept], the rewrite [erase_expr], and [drop_statics],
   which is the two together; and the condition on the environments
   under which that rewrite is sound ([envs_free_of]).

   Two passes go through a term carrying what they know of its names:
   FlambdaBlockLoadSpec.load_known_blocks, of the variables and symbols
   that are blocks, and FlambdaUnboxSpec.unbox_known_numbers, of those
   that are boxed or tagged numbers. What they carry holds variables,
   so a renaming acts on it ([rename_known_blocks],
   [rename_known_numbers], which are to be reviewed), and each pass
   commutes with renaming when what it knows is renamed with the term.
   As the validation runs them, with nothing known, they are rewrites
   that take a term and nothing else. A third pass is of the same
   kind: FlambdaAliasesSpec.follow_aliases, of the types library, which
   carries a typing environment of the first model of the types
   ([rename_naive_typing_env]).

   The checks of FlambdaWellFormedSpec.v, which say what a well-formed
   unit is: where static bindings may be ([statics_at_toplevel], with
   [raises_to], [handler_at_toplevel] and that file's
   [free_continuations]), the static names bound ([static_names]), the
   sets of closures ([sets_well_formed]), the numbers
   ([numbers_in_range]), the exception handlers
   ([unit_exn_handlers_well_formed], with [exn_handlers_well_formed],
   [ways_out_well_formed] and the three tests of how a term uses a
   continuation), the trap actions ([traps_well_formed], with
   [trap_demands], which carries the stack of the handlers installed
   and answers demands on continuations: a renaming acts on the two,
   [rename_trap_stack] and [rename_demands], which are to be
   reviewed), and the check itself ([well_formed_unit]),
   given the two continuations of the unit, which are renamed with it
   ([rename_unit_interface]); and the test of the states that the
   machine makes undefined ([static_fault]). The three conditions of
   [well_formed_unit] on binders are those of FlambdaBindersSpec.v,
   which FlambdaAlphaSpec.v says are unchanged by a renaming.

   The check of kinds of FlambdaWellKindedSpec.v, which the test tool
   runs on a unit before and after Simplify: the kind that is found
   for a [Simple] and for a defining expression, the checks of a
   static constant, of an application of a continuation and of a call,
   the check of a term ([well_kinded]) and that of a unit
   ([well_kinded_unit]). The check is given the kinds of the variables
   and the kinds of the parameters of the continuations in scope,
   which hold names, so a renaming acts on them ([rename_kind_env],
   [rename_arity_env], which are to be reviewed).

   The passes for loops, which the normalisations apply:
   FlambdaRecHandlersSpec.rewrite_rec_handlers, which takes a rewrite
   into the handlers of recursive continuations and commutes when the
   rewrite does; FlambdaSinkContsSpec.sink_let_conts;
   FlambdaShortcutSpec.shortcut_switches;
   FlambdaSameHandlersSpec.share_handlers;
   FlambdaUnusedParamsSpec.drop_unused_params;
   FlambdaDeadParamsSpec.drop_dead_params;
   FlambdaLoopAliasesSpec.follow_aliases_in_loops, which carries a
   typing environment as [follow_aliases] does;
   FlambdaLoopLetsSpec.remove_unused_defined_lets_in_loops, which
   carries the kinds of what is in scope as
   [remove_unused_defined_lets] does;
   FlambdaCodeLetsSpec.remove_unused_defined_lets_in_code, which
   starts that pass in the body of each piece of code;
   FlambdaParamAliasesSpec.bind_aliased_params;
   FlambdaForwardSpec.forward_applications, and the same with the
   calls taken on or not ([forward_applications_with]);
   FlambdaForwardJumpsSpec.forward_jumps;
   FlambdaInlineTailSpec.inline_tail_conts_kinded and
   FlambdaForwardKindedSpec.forward_kinded with
   [forward_kinded_in_code], which starts it in the body of each
   piece of code, which the validation applies to the term after; and
   FlambdaUnaliasParamsSpec.unalias_params, which takes away a
   parameter that every application gives one variable, and which the
   check applies to each term once its continuations are placed, and
   FlambdaGiveScrutineeSpec.give_scrutinees, which the check applies
   just before it and which gives such a parameter the scrutinee of a
   [Switch] where every application gives it the number of its arm;
   and FlambdaArmNumbersSpec.numbers_in_arms, which the check applies
   just after it and which puts the number of an arm for the
   scrutinee of its [Switch], and the tagged number for the variable
   that the scrutinee is the untagging of, where the arm reads them;
   and
   FlambdaWrapReturnsSpec.wrap_returns, which takes away a parameter
   that its handler does not use, of a continuation that a call
   returns to, the call returning to a wrapper, and which ends a
   round for an expression of the term before; and
   FlambdaUntagParamsSpec.untag_params, which untags, for a set of
   positions at once, the parameters of continuations that are only
   untagged or passed on, and which a round of the first phase of the
   term before applies after the unboxing; and
   FlambdaShadowParamsSpec.shadow_params, which binds again, at the
   head of their handlers, the parameters of continuations that are
   untagged, and FlambdaTwinParamsSpec.twin_params, which gives such
   a parameter a twin that holds the number, both of which a round of
   the second phase of the term before applies once it has settled
   the term, on either side of
   FlambdaRenameApartSpec.rename_apart_avoiding. And
   FlambdaBindConstParamsSpec.bind_const_params, which takes away a
   parameter that every application gives one symbol or one constant,
   as [unalias_params] does for a variable. And
   FlambdaLoopifySpec.loopify_unit, which the validation applies to
   the term before, after the first comparison and before any rewrite
   that makes names: it is given, for the code ids to make loops of,
   the continuation of each loop, and a renaming acts on those
   continuations ([rename_loopified], which is to be reviewed).

   The normalisations of FlambdaValidateSpec.v: [iterate] of a rewrite
   that commutes, the three rounds of the first phase of the term
   before ([normalise_expr_params], [normalise_code_params],
   [normalise_params_round]) and the last with the step that follows
   it ([first_phase_round]), [normalise_before_inlined],
   [normalise_after], [without_statics], [demote_and_tidy],
   [demote_and_tidy_known], and, of those that the check applies to
   the term after, [without_head_statics] and [with_code_first]. And
   the calls of the term before taken on when its rounds are over, or
   the term as it is ([with_returns]), with the test by which the
   check chooses ([keeps_a_return]): taken or not, the rewrite
   commutes with renaming and keeps scope, and the test answers the
   same of a term renamed (the last section). And [placed], which
   the check
   applies once to each term that it compares after the first
   comparison, which is of the two terms as they are given: it is
   FlambdaSinkUsesSpec.sink_conts_to_uses in the term, in the handlers
   of its recursive continuations and in the bodies of its code; it
   commutes with renaming, and it keeps scope
   ([placed_keeps_scope_statement], in the last section). And what
   the check makes of each of those terms: its continuations placed,
   then rounds of [give_scrutinees], of [unalias_params], of
   [numbers_in_arms], for the term before of
   FlambdaMergeSwitchesSpec.merge_switches, of
   FlambdaCseSpec.share_known_primitives, of following aliases and of
   removing the [Let]s left unused, and then
   FlambdaGroupsUpSpec.groups_up ([tidied_before], [tidied_after]);
   both commute with renaming, and that of the term before keeps
   scope ([tidied_before_keeps_scope_statement], in the last
   section). The check,
   [validate_unit], is in the last section. It is given the liftings
   to make
   (FlambdaLiftSpec.lifting), each of which holds the two
   continuations of the unit, so a renaming acts on it
   ([rename_lifting], which is to be reviewed); the lifting, the
   removal of unused value slots, the folding on known symbols, the
   use of the static closures in sight and the moving up of the
   bindings of code commute with renaming by the statements of
   FlambdaLiftSpec.v, of FlambdaMyClosureSpec.v, of
   FlambdaUnusedSlotsSpec.v, of FlambdaKnownSymbolsSpec.v, of
   FlambdaKnownClosuresSpec.v and of FlambdaFloatCodeSpec.v. What
   the check makes of the first three ([with_lifting],
   [without_slots], [with_known]: the rewrites where their tests
   pass, the term as it is where they do not) is stated in the
   section of the check, and beside the first the same of
   FlambdaLiftObservedSpec.with_lifting_observed, which leaves out
   the two tests of the unit, that of its exits and that of the heap,
   and which the check does not apply. Other
   rewrites that the check applies are stated to commute with
   renaming in their own files:
   FlambdaDemoteSpec.demote_handlers, of the end of the normalisation
   of the term before and of the liftings made in turn,
   FlambdaCseSpec.share_known_primitives,
   FlambdaRecoverComparisonSpec.recover_known_comparisons,
   FlambdaSameComparisonSpec.fold_same_comparisons,
   FlambdaShareProjectionsSpec.share_known_projections,
   FlambdaIdentitiesSpec.apply_identities,
   FlambdaMergeSwitchesSpec.merge_switches and
   FlambdaInlineSiblingsSpec.inline_sibling_calls, of a round of the
   term before, FlambdaMergeTablesSpec.merge_tables, which a round of
   the first phase of the term before applies first,
   FlambdaUnboxVariantsSpec.unbox_variants_knowing,
   FlambdaUnboxLoopsSpec.unbox_loops_knowing and
   FlambdaUnboxTogetherSpec.unbox_together_knowing, of
   the first phase of the term before, which are given what the static
   bindings in scope make known of their symbols, in code and in
   loops (FlambdaUnboxVariantsRenameSpec.v; what is known of symbols
   alone is not changed by a renaming, and the two functions that
   take it into code and into loops commute with renaming where the
   rewrite that they are given does: FlambdaUnboxInScopeRenameSpec.v),
   and FlambdaStaticsOrderSpec.order_statics_as, which
   the check applies to the term before with the term after as its
   pattern. One of a round of the term after is stated here:
   FlambdaUnusedProjectionsSpec.remove_unused_projections, which
   carries what it knows of the variables that are sets of closures;
   the action of a renaming on that is in its file
   ([rename_known_sets]). The rewrites from the types, which the
   normalisation of the term before applies once, between the rounds
   of its second phase and those rounds once more, commute with
   renaming by the statement of the types library
   (FlambdaSimplifyPassRenamingSpec.simplify_by_types_equivariant_statement);
   the function that the normalisation calls for them is equal to the
   one of that statement
   (FlambdaSimplifyCarriedSpec.simplify_by_types_carried_eq_statement);
   they are applied under a test that what they give keeps scope
   (FlambdaGuardScopeSpec.guard_scope), which keeps scope whatever
   the rewrite is and commutes with renaming where the rewrite does
   (FlambdaValidateScopeSpec.v). The code whose calls the check is
   told not to inline is named by code ids, which no renaming changes.

   Rewrites that make names. A rewrite that makes names with
   FlambdaNameAbstractionSpec.fresh_name, outside the names of the term
   it is given, does not commute with a renaming: which name is made
   depends on the numbers of the names of the term, and a renaming
   changes those. The section of that name says what such a rewrite
   can do instead, which is to commute up to the names that it makes
   ([equivariant_up_to]); that this composes with the rewrites above,
   for rewrites that keep scope ([keeps_scope]); and that the
   comparison of FlambdaCompareSpec.v, which is up to bound names,
   answers the same ([expr_eqb_up_to_statement]). The three
   definitions are to be reviewed. They speak of the free names by the
   one function for each sort of FlambdaBindersSpec.v, which goes into
   code.

   The validation applies one such rewrite,
   FlambdaRenameApartSpec.rename_apart_avoiding, which is given
   continuations that the names it makes are to avoid; [rename_apart]
   is that rewrite given none. After a lifting it is given the
   continuations of the units of the liftings
   (FlambdaValidateSpec.unit_continuations): a lifting reads those of
   its unit, and the term that it is applied to need not mention them,
   so a binder that was given one of them as its name could change
   what the next lifting does. Of a rewrite that is given
   continuations to avoid, commuting up to the names that it makes
   says one thing more ([equivariant_up_to_avoiding]): the
   continuations are renamed with the term, and the second renaming
   does to them what the first does ([rename_conts_alike]). Those two
   definitions are to be reviewed too.

   The section after it, the last, says what follows for the
   validation: a round of the term before, [normalise_before], and the
   normalisation of a term in which something was lifted
   ([normalise_lifted]) keep scope and commute with renaming up to the
   names that they make, the last with the continuations that it is
   given to avoid; so do the liftings made in turn, each followed by
   that normalisation ([lifted_in_turn]), where the continuations to
   avoid have those of the units of the liftings among them; the
   putting first of the static bindings ([with_statics_first]), which
   reads the continuations of the unit of a lifting, commutes with a
   renaming of the term where the liftings are under another that
   does to those continuations what it does; and [validate_unit]
   answers the same on a pair of
   terms and on the pair renamed, where the code of the term before is
   closed, and more generally where the term before is under a second
   renaming that does to its free names, and to the continuations
   that the validation is given, what the first does
   ([validate_unit_renamed_alike_statement]). They rest on what the
   file of [rename_apart] states and
   proves, that [rename_apart_avoiding] commutes with renaming up to
   the names that it makes, the second renaming doing to the
   continuations to avoid what the first does
   (FlambdaRenameApartSpec.rename_apart_avoiding_equivariant_up_to_statement),
   and on the statements that each rewrite of the validation keeps
   scope: FlambdaKeepsScopeSpec.v and FlambdaValidateScopeSpec.v for
   those of a round, and FlambdaStaticsScopeSpec.v for those that the
   check applies to the term before besides.

   Nothing is trusted here. The proofs are in
   FlambdaValidatorEquivarianceProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaNominalSpec FlambdaAlphaSpec
  FlambdaNormaliseSpec FlambdaEquivarianceSpec FlambdaUnusedContsSpec
  FlambdaUnusedStaticsSpec FlambdaBlockLoadSpec FlambdaUnboxSpec
  FlambdaBindersSpec FlambdaWellFormedSpec FlambdaWellKindedSpec
  FlambdaLiftSpec FlambdaCompareSpec FlambdaRecHandlersSpec
  FlambdaSinkContsSpec FlambdaSinkUsesSpec FlambdaShortcutSpec
  FlambdaSameHandlersSpec FlambdaUnusedParamsSpec
  FlambdaLoopAliasesSpec FlambdaLoopLetsSpec FlambdaCodeLetsSpec
  FlambdaParamAliasesSpec
  FlambdaForwardSpec FlambdaForwardJumpsSpec FlambdaForwardKindedSpec
  FlambdaForwardArmSpec
  FlambdaInlineTailSpec FlambdaLoopifySpec
  FlambdaDeadParamsSpec FlambdaRenameApartSpec FlambdaUnusedProjectionsSpec
  FlambdaReadsAfterSpec
  FlambdaUnaliasParamsSpec FlambdaWrapReturnsSpec FlambdaUntagParamsSpec
  FlambdaGiveScrutineeSpec FlambdaArmNumbersSpec FlambdaShadowParamsSpec
  FlambdaTwinParamsSpec FlambdaValidateSpec.
From OxCamlFlambda2 Require FlambdaBindConstParamsSpec.
From OxCamlFlambda2 Require Import FlambdaUnaliasRecParamsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypesSpec FlambdaAliasesSpec.
From OxCamlFlambda2 Require Import FlambdaLiftObservedSpec.
Import ListNotations.

(* A rewrite of terms that takes nothing but the term commutes with
   renaming. *)
Definition rewrite_equivariant (f : expr -> expr) : Prop :=
  ∀ (r : renaming) (e : expr), f (rename_expr r e) = rename_expr r (f e).

(* ======================================================================== *)
(* The checks of FlambdaNormaliseSpec.v                                     *)
(* ======================================================================== *)

(* The names in scope are renamed with the term. *)
Definition expr_in_scope_equivariant_statement : Prop :=
  ∀ (r : renaming) (vs : list variable) (ks : list continuation) (e : expr),
    expr_in_scope (map (rename_variable r) vs)
      (map (rename_continuation r) ks) (rename_expr r e)
    = expr_in_scope vs ks e.

(* Whether a call may be inlined, the call and the code being renamed
   together. *)
Definition inlinable_b_equivariant_statement : Prop :=
  ∀ (r : renaming) (ap : apply_expr) (c : code),
    inlinable_b (rename_apply r ap) (rename_code r c) = inlinable_b ap c.

(* The same for a call that gives tupled code a tuple. *)
Definition tupled_inlinable_b_equivariant_statement : Prop :=
  ∀ (r : renaming) (ap : apply_expr) (c : code),
    tupled_inlinable_b (rename_apply r ap) (rename_code r c)
    = tupled_inlinable_b ap c.

Definition var_free_b_equivariant_statement : Prop :=
  ∀ (r : renaming) (x : variable) (e : expr),
    var_free_b (rename_variable r x) (rename_expr r e) = var_free_b x e.

(* ======================================================================== *)
(* The rewrites of FlambdaNormaliseSpec.v                                   *)
(* ======================================================================== *)

Definition remove_unused_lets_equivariant_statement : Prop :=
  rewrite_equivariant remove_unused_lets.

(* The continuation, the parameters and the handler are renamed with the
   body; where there is no result for the one there is none for the
   other. The kinds of the parameters have no names in them and stay. *)
Definition inline_cont_at_tail_equivariant_statement : Prop :=
  ∀ (r : renaming) (k : continuation) (params : list variable)
    (metadata : handler_metadata) (handler body : expr),
    inline_cont_at_tail (rename_continuation r k)
      (map (rename_variable r) params) metadata (rename_expr r handler)
      (rename_expr r body)
    = option_map (rename_expr r)
        (inline_cont_at_tail k params metadata handler body).

Definition inline_tail_conts_equivariant_statement : Prop :=
  rewrite_equivariant inline_tail_conts.

Definition drop_deleted_code_equivariant_statement : Prop :=
  rewrite_equivariant drop_deleted_code.

Definition remove_unused_total_lets_equivariant_statement : Prop :=
  rewrite_equivariant remove_unused_total_lets.

(* The variables in scope are renamed with the term; the kinds of their
   binders have no names in them and stay
   (FlambdaNormaliseSpec.rename_kinds_in_scope). *)
Definition remove_unused_defined_lets_equivariant_statement : Prop :=
  ∀ (r : renaming) (kinds : kinds_in_scope) (e : expr),
    remove_unused_defined_lets (rename_kinds_in_scope r kinds)
      (rename_expr r e)
    = rename_expr r (remove_unused_defined_lets kinds e).

(* The removal of the projections of function slots that nothing uses
   (FlambdaUnusedProjectionsSpec.v): what it knows is renamed with the
   term. Then the pass as the validation runs it, with nothing known. *)
Definition remove_projections_knowing_equivariant_statement : Prop :=
  ∀ (r : renaming) (known : known_sets) (e : expr),
    remove_projections_knowing (rename_known_sets r known) (rename_expr r e)
    = rename_expr r (remove_projections_knowing known e).

Definition remove_unused_projections_equivariant_statement : Prop :=
  ∀ (r : renaming) (e : expr),
    remove_unused_projections (rename_expr r e)
    = rename_expr r (remove_unused_projections e).

(* The test that decides whether a call agrees with the code called in
   its arities. It reads arities, which have no names in them, and how
   many parameters the code has. *)
Definition call_agrees_b_equivariant_statement : Prop :=
  ∀ (r : renaming) (ap : apply_expr) (c : code),
    call_agrees_b (rename_apply r ap) (rename_code r c) = call_agrees_b ap c.

(* The code known is renamed with the term, as for [inline_calls]. What
   is known of the symbols holds no variable and no continuation, and
   the code whose calls are left is named by code ids, which no renaming
   changes. *)
Definition inline_known_calls_equivariant_statement : Prop :=
  ∀ (r : renaming) (kept : list code_id) (known_code : list (code_id * code))
    (known_symbols : list (symbol * option code_id)) (e : expr),
    inline_known_calls kept (map_values (rename_code r) known_code)
      known_symbols (rename_expr r e)
    = rename_expr r (inline_known_calls kept known_code known_symbols e).

Definition merge_statics_equivariant_statement : Prop :=
  rewrite_equivariant merge_statics.

Definition collapse_switches_equivariant_statement : Prop :=
  rewrite_equivariant collapse_switches.

Definition thread_switches_equivariant_statement : Prop :=
  rewrite_equivariant thread_switches.

(* A rewrite taken into the bodies of code. *)
Definition rewrite_code_equivariant_statement : Prop :=
  ∀ f : expr -> expr,
    rewrite_equivariant f -> rewrite_equivariant (rewrite_code f).

(* ======================================================================== *)
(* FlambdaUnusedContsSpec.v                                                 *)
(* ======================================================================== *)

Definition cont_free_b_equivariant_statement : Prop :=
  ∀ (r : renaming) (k : continuation) (e : expr),
    cont_free_b (rename_continuation r k) (rename_expr r e) = cont_free_b k e.

Definition remove_unused_conts_equivariant_statement : Prop :=
  rewrite_equivariant remove_unused_conts.

(* ======================================================================== *)
(* FlambdaUnusedStaticsSpec.v                                               *)
(* ======================================================================== *)

(* The names removed are symbols and code ids, which no renaming moves:
   the argument that holds them is as it was. *)
Definition expr_avoids_equivariant_statement : Prop :=
  ∀ (r : renaming) (D : dropped) (e : expr),
    expr_avoids D (rename_expr r e) = expr_avoids D e.

Definition erase_expr_equivariant_statement : Prop :=
  ∀ D : dropped, rewrite_equivariant (erase_expr D).

(* Where the names given may not be removed from the one term they may
   not be from the other. *)
Definition drop_statics_equivariant_statement : Prop :=
  ∀ (r : renaming) (D : dropped) (e : expr),
    drop_statics D (rename_expr r e)
    = option_map (rename_expr r) (drop_statics D e).

Definition binds_only_kept_equivariant_statement : Prop :=
  ∀ (r : renaming) (D : dropped) (e : expr),
    binds_only_kept D (rename_expr r e) = binds_only_kept D e.

(* The condition on the environments under which removing static
   bindings is sound (the premise of
   FlambdaUnusedStaticsSpec.drop_statics_refines_statement), with the
   action of FlambdaEquivarianceSpec.v on what a state holds. *)
Definition envs_free_of_equivariant_statement : Prop :=
  ∀ (r : renaming) (D : dropped) (vars : env) (conts : cont_env)
    (statics : static_env),
    envs_free_of D (rename_env r vars) (rename_cont_env r conts)
      (rename_static_env r statics)
    = envs_free_of D vars conts statics.

(* ======================================================================== *)
(* FlambdaBlockLoadSpec.v: what is known of blocks                          *)
(* ======================================================================== *)

(* The pass carries what is known of the blocks of the term, which holds
   variables: those that are known to be blocks, and those that the
   fields of a block are known to hold. A renaming acts on both. The
   symbols that are known to be blocks, the tags and the numbers of
   fields are as they were. *)
Definition rename_known_block (r : renaming) (block : known_block)
    : known_block :=
  (fst block, map (option_map (rename_simple r)) (snd block)).

Definition rename_known_blocks (r : renaming) (known : known_blocks)
    : known_blocks :=
  Known_blocks
    (map
       (fun entry =>
          (rename_variable r (fst entry), rename_known_block r (snd entry)))
       (known_vars known))
    (map_values (rename_known_block r) (known_symbols known)).

(* What is known is renamed with the term. *)
Definition load_blocks_knowing_equivariant_statement : Prop :=
  ∀ (r : renaming) (known : known_blocks) (e : expr),
    load_blocks_knowing (rename_known_blocks r known) (rename_expr r e)
    = rename_expr r (load_blocks_knowing known e).

(* The pass as the validation runs it, with nothing known. *)
Definition load_known_blocks_equivariant_statement : Prop :=
  rewrite_equivariant load_known_blocks.

(* The same of the read of a known block that the term after is given
   (FlambdaReadsAfterSpec.load_known_blocks_surely). *)
Definition load_blocks_knowing_surely_equivariant_statement : Prop :=
  ∀ (r : renaming) (known : known_blocks) (e : expr),
    load_blocks_knowing_surely (rename_known_blocks r known) (rename_expr r e)
    = rename_expr r (load_blocks_knowing_surely known e).

Definition load_known_blocks_surely_equivariant_statement : Prop :=
  rewrite_equivariant load_known_blocks_surely.

(* ======================================================================== *)
(* FlambdaUnboxSpec.v: what is known of wrapped numbers                     *)
(* ======================================================================== *)

(* Likewise: the variables that are known to be wrapped numbers are
   renamed, and what each wraps. What is known of a symbol is a
   constant. *)
Definition rename_known_numbers (r : renaming) (known : known_numbers)
    : known_numbers :=
  Known_numbers
    (map
       (fun entry =>
          (rename_variable r (fst entry),
           (fst (snd entry), rename_simple r (snd (snd entry)))))
       (known_wrapped_vars known))
    (known_boxed_symbols known).

Definition unbox_numbers_knowing_equivariant_statement : Prop :=
  ∀ (r : renaming) (known : known_numbers) (e : expr),
    unbox_numbers_knowing (rename_known_numbers r known) (rename_expr r e)
    = rename_expr r (unbox_numbers_knowing known e).

Definition unbox_known_numbers_equivariant_statement : Prop :=
  rewrite_equivariant unbox_known_numbers.

(* ======================================================================== *)
(* The types library: following aliases                                     *)
(* ======================================================================== *)

(* FlambdaAliasesSpec.follow_aliases carries a typing environment of the
   first model of the types (FlambdaTypesSpec.v): variables, each with
   what is known of it, which may hold variables. A renaming acts on
   both. *)
Definition rename_naive_type (r : renaming) (t : flambda_type)
    : flambda_type :=
  match t with
  | Unknown => Unknown
  | Bottom => Bottom
  | Equals s => Equals (rename_simple r s)
  | Closure_of code value_slots =>
    Closure_of code
      (map (fun slot => (fst slot, rename_simple r (snd slot))) value_slots)
  end.

Definition rename_naive_typing_env (r : renaming) (tenv : typing_env)
    : typing_env :=
  map
    (fun entry =>
       (rename_variable r (fst entry), rename_naive_type r (snd entry)))
    tenv.

Definition follow_aliases_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : typing_env) (e : expr),
    follow_aliases (rename_naive_typing_env r tenv) (rename_expr r e)
    = rename_expr r (follow_aliases tenv e).

(* The pass as the validation runs it, with nothing known. *)
Definition follow_aliases_unknowing_equivariant_statement : Prop :=
  rewrite_equivariant (follow_aliases []).

(* ======================================================================== *)
(* The checks of FlambdaWellFormedSpec.v                                    *)
(* ======================================================================== *)

(* The continuations that a term may apply and does not bind, as that
   file computes them: those of the renamed term are the images of
   those of the term, in the same order. *)
Definition well_formed_free_continuations_equivariant_statement : Prop :=
  ∀ (r : renaming) (e : expr),
    free_continuations (rename_expr r e)
    = map (rename_continuation r) (free_continuations e).

Definition raises_to_equivariant_statement : Prop :=
  ∀ (r : renaming) (k : continuation) (e : expr),
    raises_to (rename_continuation r k) (rename_expr r e) = raises_to k e.

(* Where static bindings may be. The exception continuation of the unit
   is renamed with the term. Whether a handler says that it is an
   exception handler is in its metadata, which a renaming keeps. *)
Definition handler_at_toplevel_equivariant_statement : Prop :=
  ∀ (r : renaming) (unit_exn k : continuation) (h : continuation_handler)
    (body : expr),
    handler_at_toplevel (rename_continuation r unit_exn)
      (rename_continuation r k) (rename_handler r h) (rename_expr r body)
    = handler_at_toplevel unit_exn k h body.

Definition statics_at_toplevel_equivariant_statement : Prop :=
  ∀ (r : renaming) (unit_exn : continuation) (toplevel : bool) (e : expr),
    statics_at_toplevel (rename_continuation r unit_exn) toplevel
      (rename_expr r e)
    = statics_at_toplevel unit_exn toplevel e.

(* The static names that a term binds are those of the renamed term,
   for a way of reading the names of a binding that does not look at
   its variables and continuations. *)
Definition static_names_equivariant_statement : Prop :=
  ∀ (r : renaming) (names_of : static_binding -> list positive) (e : expr),
    (∀ binding : static_binding,
      names_of (rename_static_binding r binding) = names_of binding) ->
    static_names names_of (rename_expr r e) = static_names names_of e.

(* So the code ids and the symbols that a term binds, which are the two
   ways used, are those of the renamed term. *)
Definition bound_static_names_equivariant_statement : Prop :=
  ∀ (r : renaming) (e : expr),
    static_names code_ids_of_binding (rename_expr r e)
    = static_names code_ids_of_binding e
    ∧ static_names symbols_of_binding (rename_expr r e)
      = static_names symbols_of_binding e.

Definition sets_well_formed_equivariant_statement : Prop :=
  ∀ (r : renaming) (e : expr),
    sets_well_formed (rename_expr r e) = sets_well_formed e.

Definition numbers_in_range_equivariant_statement : Prop :=
  ∀ (r : renaming) (e : expr),
    numbers_in_range (rename_expr r e) = numbers_in_range e.

(* Exception handlers are apart from the other continuations: the
   tests of how a term uses a continuation ([applies_cont],
   [gives_extra_args], [installs_or_removes]), the check of the
   continuations that code or a unit does not bind
   ([ways_out_well_formed]), and the check of a term and of a unit. *)
Definition applies_cont_equivariant_statement : Prop :=
  ∀ (r : renaming) (k : continuation) (e : expr),
    applies_cont (rename_continuation r k) (rename_expr r e)
    = applies_cont k e.

Definition gives_extra_args_equivariant_statement : Prop :=
  ∀ (r : renaming) (k : continuation) (e : expr),
    gives_extra_args (rename_continuation r k) (rename_expr r e)
    = gives_extra_args k e.

Definition installs_or_removes_equivariant_statement : Prop :=
  ∀ (r : renaming) (k : continuation) (e : expr),
    installs_or_removes (rename_continuation r k) (rename_expr r e)
    = installs_or_removes k e.

Definition ways_out_well_formed_equivariant_statement : Prop :=
  ∀ (r : renaming) (exn : continuation) (e : expr),
    ways_out_well_formed (rename_continuation r exn) (rename_expr r e)
    = ways_out_well_formed exn e.

Definition exn_handlers_well_formed_equivariant_statement : Prop :=
  ∀ (r : renaming) (e : expr),
    exn_handlers_well_formed (rename_expr r e) = exn_handlers_well_formed e.

Definition unit_exn_handlers_well_formed_equivariant_statement : Prop :=
  ∀ (r : renaming) (unit_exn : continuation) (e : expr),
    unit_exn_handlers_well_formed (rename_continuation r unit_exn)
      (rename_expr r e)
    = unit_exn_handlers_well_formed unit_exn e.

(* The discipline of trap actions. The check follows a term with the
   stack of the exception handlers installed, and answers what the term
   asks of the continuations that it does not bind: each one it may go
   to, with the stack when it gets there. Both hold names of
   continuations. The action on the two: every continuation is
   renamed, and the order of a stack and of the demands is kept. *)
Definition rename_trap_stack (r : renaming) (ts : trap_stack) : trap_stack :=
  map (rename_continuation r) ts.

Definition rename_demands (r : renaming) (ds : demands) : demands :=
  map (fun d => (rename_continuation r (fst d), rename_trap_stack r (snd d)))
    ds.

(* The demands of the renamed term under the renamed stack are those of
   the term, renamed; where the term does not follow the discipline,
   the renamed term does not either. *)
Definition trap_demands_equivariant_statement : Prop :=
  ∀ (r : renaming) (ts : trap_stack) (e : expr),
    trap_demands (rename_trap_stack r ts) (rename_expr r e)
    = option_map (rename_demands r) (trap_demands ts e).

Definition traps_well_formed_equivariant_statement : Prop :=
  ∀ (r : renaming) (unit_exn : continuation) (e : expr),
    traps_well_formed (rename_continuation r unit_exn) (rename_expr r e)
    = traps_well_formed unit_exn e.

(* The action on what a unit is given by where it is run: its two
   continuations are renamed. *)
Definition rename_unit_interface (r : renaming) (u : unit_interface)
    : unit_interface :=
  Unit_interface (rename_continuation r (unit_return u))
    (rename_continuation r (unit_exn u)).

(* A renamed unit is well-formed when the unit is, what it is given
   being renamed with it. Three of the conditions are those of
   FlambdaBindersSpec.v, whose statements are in FlambdaAlphaSpec.v. *)
Definition well_formed_unit_equivariant_statement : Prop :=
  ∀ (r : renaming) (u : unit_interface) (e : expr),
    well_formed_unit (rename_unit_interface r u) (rename_expr r e)
    = well_formed_unit u e.

(* The test of the states that the machine makes undefined to keep a
   static binding from being executed twice, with the action of
   FlambdaEquivarianceSpec.v on states. *)
Definition static_fault_equivariant_statement : Prop :=
  ∀ (r : renaming) (s : state),
    static_fault (rename_state r s) = static_fault s.

(* ======================================================================== *)
(* The passes for loops                                                     *)
(* ======================================================================== *)

(* A rewrite taken into the handlers of recursive continuations
   (FlambdaRecHandlersSpec.v) commutes with renaming when the rewrite
   does. *)
Definition rewrite_rec_handlers_equivariant_statement : Prop :=
  ∀ f : expr -> expr,
    rewrite_equivariant f -> rewrite_equivariant (rewrite_rec_handlers f).

(* Moving a continuation to the handler that alone uses it
   (FlambdaSinkContsSpec.v). *)
Definition sink_let_conts_equivariant_statement : Prop :=
  rewrite_equivariant sink_let_conts.

(* Moving a continuation down past the bindings under it
   (FlambdaSinkUsesSpec.v). *)
Definition sink_conts_to_uses_equivariant_statement : Prop :=
  rewrite_equivariant sink_conts_to_uses.

(* An arm of a [Switch] that applies a continuation which only applies
   another (FlambdaShortcutSpec.v). *)
Definition shortcut_switches_equivariant_statement : Prop :=
  rewrite_equivariant shortcut_switches.

(* A continuation whose handler is that of a continuation bound around
   it applies that one (FlambdaSameHandlersSpec.v). *)
Definition share_handlers_equivariant_statement : Prop :=
  rewrite_equivariant share_handlers.

(* Removing the parameters of a continuation that its handler does not
   use (FlambdaUnusedParamsSpec.v). *)
Definition drop_unused_params_equivariant_statement : Prop :=
  rewrite_equivariant drop_unused_params.

(* Removing the parameters of a loop that are only passed round it, and
   the [Let]s that only they need (FlambdaDeadParamsSpec.v). *)
Definition drop_dead_params_equivariant_statement : Prop :=
  rewrite_equivariant drop_dead_params.

(* Following aliases into the handlers of recursive continuations
   (FlambdaLoopAliasesSpec.v), with the action on the typing
   environment that [follow_aliases] has. *)
Definition follow_aliases_in_loops_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : typing_env) (e : expr),
    follow_aliases_in_loops (rename_naive_typing_env r tenv)
      (rename_expr r e)
    = rename_expr r (follow_aliases_in_loops tenv e).

(* The pass as the validation runs it, with nothing known. *)
Definition follow_aliases_in_loops_unknowing_equivariant_statement : Prop :=
  rewrite_equivariant (follow_aliases_in_loops []).

(* Removing the unused [Let]s that are sure to have a value, in the
   handlers of recursive continuations too (FlambdaLoopLetsSpec.v):
   the kinds of what is in scope are renamed with the term. The
   validation runs it with nothing in scope. *)
Definition remove_unused_defined_lets_in_loops_equivariant_statement : Prop :=
  ∀ (r : renaming) (kinds : kinds_in_scope) (e : expr),
    remove_unused_defined_lets_in_loops (rename_kinds_in_scope r kinds)
      (rename_expr r e)
    = rename_expr r (remove_unused_defined_lets_in_loops kinds e).

(* The same pass in the body of each piece of code, started with the
   parameters of the code and [my_closure] in scope
   (FlambdaCodeLetsSpec.v). *)
Definition remove_unused_defined_lets_in_code_equivariant_statement : Prop :=
  rewrite_equivariant remove_unused_defined_lets_in_code.

(* Binding a parameter that every application gives the same thing
   (FlambdaParamAliasesSpec.v). *)
Definition bind_aliased_params_equivariant_statement : Prop :=
  rewrite_equivariant bind_aliased_params.

(* An application of a continuation that only applies another
   (FlambdaForwardSpec.v). *)
Definition forward_applications_equivariant_statement : Prop :=
  rewrite_equivariant forward_applications.

(* The same with the calls taken on or not
   (FlambdaForwardSpec.forward_applications_with). *)
Definition forward_applications_with_equivariant_statement : Prop :=
  ∀ calls : bool, rewrite_equivariant (forward_applications_with calls).

(* The same for the continuations of no parameters, where it is an
   equivalence (FlambdaForwardJumpsSpec.v). *)
Definition forward_jumps_equivariant_statement : Prop :=
  rewrite_equivariant forward_jumps.

(* The same where the arguments are known to have the kinds of the
   parameters, an equivalence too (FlambdaForwardKindedSpec.v). *)
Definition forward_kinded_equivariant_statement : Prop :=
  rewrite_equivariant forward_kinded.

(* The same pass in the body of each piece of code, started with the
   kinds of the parameters of the code and of [my_closure]. *)
Definition forward_kinded_in_code_equivariant_statement : Prop :=
  rewrite_equivariant forward_kinded_in_code.

(* An application that gives a number to the parameter that the handler
   of its continuation tests, made to go where the arm of that number
   goes (FlambdaForwardArmSpec.v): with nothing asked of the kinds,
   where the arguments are known to have the kinds of the parameters,
   and the latter in the body of each piece of code. *)
Definition forward_to_arms_equivariant_statement : Prop :=
  rewrite_equivariant forward_to_arms.

Definition forward_to_arms_kinded_equivariant_statement : Prop :=
  rewrite_equivariant forward_to_arms_kinded.

Definition forward_to_arms_in_code_equivariant_statement : Prop :=
  rewrite_equivariant forward_to_arms_in_code.

(* Inlining a continuation at the end of its scope, where its handler
   gives one kind for each parameter (FlambdaInlineTailSpec.v). *)
Definition inline_tail_conts_kinded_equivariant_statement : Prop :=
  rewrite_equivariant inline_tail_conts_kinded.

(* Taking away a parameter of a continuation that every application
   gives one variable, which is at the handler what it is at the
   applications (FlambdaUnaliasParamsSpec.v), as the check does once
   the continuations are placed. That it keeps scope is in
   FlambdaStaticsScopeSpec.v. *)
Definition unalias_params_equivariant_statement : Prop :=
  rewrite_equivariant unalias_params.

(* Giving a parameter of a continuation the scrutinee of a [Switch]
   where every application gives it the number of the arm that it is
   in (FlambdaGiveScrutineeSpec.v). The variable and the positions are
   found by tests of equality of names, and no name is made. That it
   keeps scope is in FlambdaStaticsScopeSpec.v. *)
Definition give_scrutinees_equivariant_statement : Prop :=
  rewrite_equivariant give_scrutinees.

(* Putting the number of an arm for the scrutinee of its [Switch],
   and the tagged number for the variable that the scrutinee is the
   untagging of, where the arm and the handler that only that arm
   applies read them (FlambdaArmNumbersSpec.v, which says in which
   places). The variables are found by tests of equality of names, in
   the order of the term, and no name is made. That it keeps scope is
   in FlambdaStaticsScopeSpec.v. *)
Definition numbers_in_arms_equivariant_statement : Prop :=
  rewrite_equivariant numbers_in_arms.

(* Removing a parameter that its handler does not use, of a
   continuation that a call returns to, with a wrapper for the call
   (FlambdaWrapReturnsSpec.v). The wrapper has the name of the
   continuation and the parameters of its handler, so no name is made.
   That it keeps scope is in FlambdaStaticsScopeSpec.v. *)
Definition wrap_returns_equivariant_statement : Prop :=
  rewrite_equivariant wrap_returns.

(* Untagging, for a set of positions at once, the parameters of
   continuations that hold a tagged immediate and are only untagged or
   passed on (FlambdaUntagParamsSpec.v). The positions are computed
   from the term by tests of equality of names, so those of the
   renamed term are those of the term with the continuations renamed;
   no name is made. That it keeps scope is in
   FlambdaStaticsScopeSpec.v. *)
Definition untag_params_equivariant_statement : Prop :=
  rewrite_equivariant untag_params.

(* Binding again, at the head of their handlers, the parameters of
   continuations that hold a tagged immediate and are untagged
   (FlambdaShadowParamsSpec.v). The positions are computed from the
   term by tests of equality of names; the [Let]s that are put bind
   the parameters themselves, so no name is made. That it keeps
   scope is in FlambdaStaticsScopeSpec.v. *)
Definition shadow_params_equivariant_statement : Prop :=
  rewrite_equivariant shadow_params.

(* Giving a twin that holds the number to a parameter of a
   continuation that the head of its handler binds again
   (FlambdaTwinParamsSpec.v). The positions are computed from the
   term by tests of equality of names; the name of a twin is the
   parameter's own, and the name that takes the place of the
   parameter is that of a [Let] of the term, so no name is made.
   That it keeps scope is in FlambdaStaticsScopeSpec.v. *)
Definition twin_params_equivariant_statement : Prop :=
  rewrite_equivariant twin_params.

(* Taking away a parameter of a continuation that every application
   gives one symbol or one constant (FlambdaBindConstParamsSpec.v).
   The applications are found by tests of equality of names, a
   renaming moves no symbol and no constant, and no name is made. That
   it keeps scope is in FlambdaStaticsScopeSpec.v. *)
Definition bind_const_params_equivariant_statement : Prop :=
  rewrite_equivariant FlambdaBindConstParamsSpec.bind_const_params.

(* Taking away a parameter of a loop that every turn passes on as it
   is and that the loop is given one variable for
   (FlambdaUnaliasRecParamsSpec.v). The applications are found by
   tests of equality of names, and no name is made. That it keeps
   scope is in FlambdaStaticsScopeSpec.v. *)
Definition unalias_rec_params_equivariant_statement : Prop :=
  rewrite_equivariant unalias_rec_params.

(* FlambdaLoopifySpec.loopify_unit is given, for each code id to make
   a loop of, the continuation of the loop. The action: the
   continuation is renamed, the code id is not. *)
Definition rename_loopified (r : renaming) (G : loopified) : loopified :=
  map (fun entry => (fst entry, rename_continuation r (snd entry))) G.

Definition loopify_unit_equivariant_statement : Prop :=
  ∀ (r : renaming) (G : loopified) (e : expr),
    loopify_unit (rename_loopified r G) (rename_expr r e)
    = rename_expr r (loopify_unit G e).

(* ======================================================================== *)
(* The normalisations of FlambdaValidateSpec.v                              *)
(* ======================================================================== *)

(* A rewrite that commutes with renaming does so however many times it
   is applied. *)
Definition iterate_equivariant_statement : Prop :=
  ∀ (rounds : nat) (f : expr -> expr),
    rewrite_equivariant f -> rewrite_equivariant (iterate rounds f).

(* The first phase of the term before Simplify: the round for an
   expression, for the body of code and for a term. Each is made of
   the rewrites above and of [fold_constants]; no call is inlined and
   no name is made. *)
Definition normalise_expr_params_equivariant_statement : Prop :=
  rewrite_equivariant normalise_expr_params.

Definition normalise_code_params_equivariant_statement : Prop :=
  rewrite_equivariant normalise_code_params.

Definition normalise_params_round_equivariant_statement : Prop :=
  rewrite_equivariant normalise_params_round.

(* That round and what follows it in the first phase
   ([first_phase_round]). *)
Definition first_phase_round_equivariant_statement : Prop :=
  rewrite_equivariant first_phase_round.

(* The round of the second phase of the term before Simplify once its
   calls are inlined and its binders have names of their own: it is
   made of the rewrites above and of [fold_constants]. *)
Definition normalise_before_inlined_equivariant_statement : Prop :=
  rewrite_equivariant normalise_before_inlined.

(* The normalisation of the term after Simplify: it is made of the
   rewrites above and of [fold_constants]. The code whose calls are not
   inlined is named by code ids, which no renaming changes. That of the
   term before is in the last section: a round of it makes names. *)
Definition normalise_after_equivariant_statement : Prop :=
  ∀ (rounds : nat) (kept : list code_id),
    rewrite_equivariant (normalise_after rounds kept).

(* The placing of continuations that the check makes, once, of each
   term that it compares (FlambdaValidateSpec.placed). *)
Definition placed_equivariant_statement : Prop :=
  rewrite_equivariant placed.

(* What the check makes of each term that it compares after the first
   comparison: its continuations placed, and then the rounds of
   FlambdaValidateSpec.tidy_before_round for the term before, of
   [tidy_after_round] for the term after, and then
   FlambdaGroupsUpSpec.groups_up. The rounds are made of rewrites
   above and of FlambdaCseSpec.share_known_primitives, which is
   stated to commute with renaming in its file, as [groups_up] is. *)
Definition tidied_before_equivariant_statement : Prop :=
  ∀ rounds : nat, rewrite_equivariant (tidied_before rounds).

Definition tidied_after_equivariant_statement : Prop :=
  ∀ rounds : nat, rewrite_equivariant (tidied_after rounds).

(* The removal of the static bindings of the names given, where nothing
   refers to them. *)
Definition without_statics_equivariant_statement : Prop :=
  ∀ D : dropped, rewrite_equivariant (without_statics D).

(* The same removal from the head of the term after, where the checks
   of FlambdaUnusedHeadStaticsSpec.head_statics_removable pass
   (FlambdaValidateSpec.without_head_statics): the checks read the
   same of a term renamed, as a renaming changes no symbol and no code
   id. *)
Definition without_head_statics_equivariant_statement : Prop :=
  ∀ D : dropped, rewrite_equivariant (without_head_statics D).

(* The demotion of the exception handlers that no call can raise to,
   and the rounds that follow it, with which the normalisation of the
   term before and the liftings made in turn end
   (FlambdaValidateSpec.demote_and_tidy): they make no name. *)
Definition demote_and_tidy_equivariant_statement : Prop :=
  ∀ rounds : nat, rewrite_equivariant (demote_and_tidy rounds).

(* What ends the liftings made in turn
   (FlambdaValidateSpec.demote_and_tidy_known): that demotion with its
   rounds, and then rounds that use what is known of the static
   closures in sight and of the symbols that the unit binds, and read
   the value slots that are known, each followed by the demotion with
   its rounds again. No call is inlined in them and nothing is renamed
   apart: they make no name. *)
Definition demote_and_tidy_known_equivariant_statement : Prop :=
  ∀ rounds : nat, rewrite_equivariant (demote_and_tidy_known rounds).

(* The unit with all its code bound first, which the check applies to
   the term after: the conditions of FlambdaHoistCodeSpec.hoist_code
   hold of the renamed term where they hold of the term (they go by
   symbols, code ids and where the static bindings are, and by the
   free continuations of the term, which are renamed with it), and
   the binding of the code first and the tidying of the groups of
   static bindings commute with renaming. *)
Definition with_code_first_equivariant_statement : Prop :=
  rewrite_equivariant with_code_first.

(* ======================================================================== *)
(* The check of kinds of FlambdaWellKindedSpec.v                            *)
(* ======================================================================== *)

(* The check of kinds is given the kinds of the variables in scope and
   the kinds of the parameters of the continuations in scope. The
   action on the two: each name is renamed and keeps what it is bound
   to. The kinds of the value slots are keyed by slots, which no
   renaming changes. *)
Definition rename_kind_env (r : renaming) (kinds : kind_env) : kind_env :=
  Keyed.map_keys (rename_variable r) kinds.

Definition rename_arity_env (r : renaming) (conts : arity_env) : arity_env :=
  Keyed.map_keys (rename_continuation r) conts.

(* The kinds of the parameters of a handler have no names in them. *)
Definition handler_kinds_equivariant_statement : Prop :=
  ∀ (r : renaming) (h : continuation_handler),
    handler_kinds (rename_handler r h) = handler_kinds h.

(* The kind that is found for a [Simple], for the arguments of a
   primitive or of a call, and for a defining expression. *)
Definition kind_of_simple_equivariant_statement : Prop :=
  ∀ (r : renaming) (kinds : kind_env) (s : simple),
    kind_of_simple (rename_kind_env r kinds) (rename_simple r s)
    = kind_of_simple kinds s.

Definition kinds_of_simples_equivariant_statement : Prop :=
  ∀ (r : renaming) (kinds : kind_env) (args : list simple),
    kinds_of_simples (rename_kind_env r kinds) (map (rename_simple r) args)
    = kinds_of_simples kinds args.

Definition kind_of_named_equivariant_statement : Prop :=
  ∀ (r : renaming) (slots : slot_kinds) (kinds : kind_env) (d : named),
    kind_of_named slots (rename_kind_env r kinds) (rename_named r d)
    = kind_of_named slots kinds d.

(* The kinds of parameters are bound for the renamed parameters as for
   the parameters. *)
Definition bind_param_kinds_equivariant_statement : Prop :=
  ∀ (r : renaming) (params : list variable) (params_kinds : list flambda_kind)
    (kinds : kind_env),
    bind_param_kinds (map (rename_variable r) params) params_kinds
      (rename_kind_env r kinds)
    = option_map (rename_kind_env r)
        (bind_param_kinds params params_kinds kinds).

(* The checks of a static constant, of an application of a
   continuation and of a call. *)
Definition static_const_well_kinded_equivariant_statement : Prop :=
  ∀ (r : renaming) (slots : slot_kinds) (kinds : kind_env)
    (const : static_const),
    static_const_well_kinded slots (rename_kind_env r kinds)
      (rename_static_const r const)
    = static_const_well_kinded slots kinds const.

Definition apply_cont_well_kinded_equivariant_statement : Prop :=
  ∀ (r : renaming) (kinds : kind_env) (conts : arity_env) (ac : apply_cont),
    apply_cont_well_kinded (rename_kind_env r kinds) (rename_arity_env r conts)
      (rename_apply_cont r ac)
    = apply_cont_well_kinded kinds conts ac.

Definition apply_well_kinded_equivariant_statement : Prop :=
  ∀ (r : renaming) (kinds : kind_env) (conts : arity_env) (ap : apply_expr),
    apply_well_kinded (rename_kind_env r kinds) (rename_arity_env r conts)
      (rename_apply r ap)
    = apply_well_kinded kinds conts ap.

(* The check of a term, what is in scope being renamed with the term;
   and the check of a unit, its two continuations being renamed with
   it. *)
Definition well_kinded_equivariant_statement : Prop :=
  ∀ (r : renaming) (slots : slot_kinds) (kinds : kind_env) (conts : arity_env)
    (e : expr),
    well_kinded slots (rename_kind_env r kinds) (rename_arity_env r conts)
      (rename_expr r e)
    = well_kinded slots kinds conts e.

Definition well_kinded_unit_equivariant_statement : Prop :=
  ∀ (r : renaming) (slots : slot_kinds) (unit_return unit_exn : continuation)
    (e : expr),
    well_kinded_unit slots (rename_continuation r unit_return)
      (rename_continuation r unit_exn) (rename_expr r e)
    = well_kinded_unit slots unit_return unit_exn e.

(* ======================================================================== *)
(* The check                                                                *)
(* ======================================================================== *)

(* The action on what is to be lifted: the two continuations of the unit
   are renamed. A [Let] to lift is named by where it is, and what it is
   lifted to by symbols and code ids; a group to put is named by where
   it is put and by the [Let]s that it stands for, and is read off the
   term (FlambdaLiftAroundSpec.groups_of); a table to bind is a symbol
   with constants and symbols, and one that has a variable is not
   bound (FlambdaUnusedHeadStaticsSpec.add_head_tables); a copy of
   code is named by code ids and symbols
   (FlambdaCopyCodeSpec.copy_static_code); whether the calls of the
   closures that a [Let] makes are made direct in the rounds that
   follow has no name in it, nor has whether what [my_closure] is
   known to be is used in them. *)
Definition rename_lifting (r : renaming) (L : lifting) : lifting :=
  Lifting (rename_unit_interface r (lifting_unit L)) (lifted_lets L)
    (lifted_my_closures L) (lifted_out_of_code L) (lifted_to_head L)
    (lifted_around L) (groups_put L) (tables_put L) (code_copied L)
    (made_closures_called L) (own_closures_known L).

(* What the check makes of a lifting, of the value slots to remove and
   of what the unit binds its symbols to
   (FlambdaValidateSpec.with_lifting, [without_slots], [with_known]):
   each applies rewrites of other files where their tests pass, and
   leaves the term as it is where they do not. None makes a name, and
   each commutes with renaming; that they keep scope is in
   FlambdaStaticsScopeSpec.v. The lifting is renamed with the term:
   its tests go by the two continuations of its unit. The value slots
   are not renamed. *)
Definition with_lifting_equivariant_statement : Prop :=
  ∀ (r : renaming) (L : lifting) (e : expr),
    with_lifting (rename_lifting r L) (rename_expr r e)
    = rename_expr r (with_lifting L e).

(* The same of the lifting without the two tests of the unit
   (FlambdaLiftObservedSpec.with_lifting_observed): its other tests go
   by the two continuations of its unit, as those of [with_lifting]
   do. *)
Definition with_lifting_observed_equivariant_statement : Prop :=
  ∀ (r : renaming) (L : lifting) (e : expr),
    with_lifting_observed (rename_lifting r L) (rename_expr r e)
    = rename_expr r (with_lifting_observed L e).

Definition without_slots_equivariant_statement : Prop :=
  ∀ slots : list value_slot, rewrite_equivariant (without_slots slots).

Definition with_known_equivariant_statement : Prop :=
  rewrite_equivariant with_known.

(* That the validation answers the same on two terms and on the two
   renamed is in the last section: a round of the term before makes
   names. So are the liftings made in turn, which have those rounds
   between them, and the putting first of the static bindings, which
   the check applies to a term that is under a renaming of its own by
   then. *)

(* ======================================================================== *)
(* Rewrites that make names                                                 *)
(* ======================================================================== *)

(* Two renamings rename the free variables and the free continuations
   of a term alike. The free names are those of the one function for
   each sort, which goes into code. *)
Definition renames_alike (r r' : renaming) (e : expr) : Prop :=
  (∀ x : variable,
    among x (free_variables_of_expr e) = true ->
    rename_variable r' x = rename_variable r x)
  ∧ (∀ k : continuation,
      among k (free_continuations_of_expr e) = true ->
      rename_continuation r' k = rename_continuation r k).

(* A rewrite commutes with renaming up to the names that it makes: on a
   term whose code is closed, the result for the renamed term is the
   result for the term under a renaming that renames the free names of
   that result as the first renaming does. The two renamings may differ
   on the names that are bound in the result, and on names that are not
   in it. *)
Definition equivariant_up_to (f : expr -> expr) : Prop :=
  ∀ (r : renaming) (e : expr),
    code_closed e = true ->
    ∃ r' : renaming,
      renames_alike r r' (f e) ∧ f (rename_expr r e) = rename_expr r' (f e).

(* A rewrite keeps scope: on a term whose code is closed, the code of
   the result is closed, and a variable or a continuation that is free
   in the result is free in the term. Symbols and code ids are not
   counted: no renaming moves one, and a rewrite that keeps scope may
   mention or bind one that the term does not. *)
Definition keeps_scope (f : expr -> expr) : Prop :=
  ∀ e : expr,
    code_closed e = true ->
    code_closed (f e) = true
    ∧ (∀ x : variable,
        among x (free_variables_of_expr (f e)) = true ->
        among x (free_variables_of_expr e) = true)
    ∧ (∀ k : continuation,
        among k (free_continuations_of_expr (f e)) = true ->
        among k (free_continuations_of_expr e) = true).

(* A rewrite that commutes with renaming does so up to the names that
   it makes: the second renaming may be taken to be the first. *)
Definition rewrite_equivariant_up_to_statement : Prop :=
  ∀ f : expr -> expr, rewrite_equivariant f -> equivariant_up_to f.

(* Two rewrites that keep scope, one after the other, keep scope; and
   if each commutes with renaming up to the names that it makes, so
   does the one after the other. *)
Definition keeps_scope_compose_statement : Prop :=
  ∀ f g : expr -> expr,
    keeps_scope f -> keeps_scope g -> keeps_scope (fun e => g (f e)).

Definition equivariant_up_to_compose_statement : Prop :=
  ∀ f g : expr -> expr,
    keeps_scope f ->
    keeps_scope g ->
    equivariant_up_to f ->
    equivariant_up_to g ->
    equivariant_up_to (fun e => g (f e)).

(* The same of a rewrite applied a number of times. *)
Definition keeps_scope_iterate_statement : Prop :=
  ∀ (rounds : nat) (f : expr -> expr),
    keeps_scope f -> keeps_scope (iterate rounds f).

Definition equivariant_up_to_iterate_statement : Prop :=
  ∀ (rounds : nat) (f : expr -> expr),
    keeps_scope f -> equivariant_up_to f ->
    equivariant_up_to (iterate rounds f).

(* The comparison is up to bound names, so it does not tell two
   renamings of its first term apart that rename the free names of
   that term alike: it answers as it does of the two terms that are
   not renamed. *)
Definition expr_eqb_renamed_alike_statement : Prop :=
  ∀ (r r' : renaming) (e e' : expr),
    code_closed e = true ->
    renames_alike r r' e ->
    expr_eqb [] [] (rename_expr r' e) (rename_expr r e')
    = expr_eqb [] [] e e'.

(* So where the first term of a comparison is the result of a rewrite
   that keeps scope and commutes with renaming up to the names that it
   makes, the comparison answers the same when the term rewritten and
   the second term are renamed. *)
Definition expr_eqb_up_to_statement : Prop :=
  ∀ (f : expr -> expr) (r : renaming) (e e' : expr),
    keeps_scope f ->
    equivariant_up_to f ->
    code_closed e = true ->
    expr_eqb [] [] (f (rename_expr r e)) (rename_expr r e')
    = expr_eqb [] [] (f e) e'.

(* Two renamings rename the continuations of a list alike. *)
Definition rename_conts_alike (r r' : renaming) (ks : list continuation)
    : Prop :=
  ∀ k : continuation,
    among k ks = true -> rename_continuation r' k = rename_continuation r k.

(* A rewrite that is given continuations that the names it makes are to
   avoid commutes with renaming up to the names that it makes: as
   [equivariant_up_to], the continuations being renamed with the term,
   and the second renaming does to those continuations what the first
   does. The caller holds them and the term need not mention them, so
   what the second renaming does to the free names of the result says
   nothing of them. *)
Definition equivariant_up_to_avoiding
    (f : list continuation -> expr -> expr) : Prop :=
  ∀ (r : renaming) (avoid : list continuation) (e : expr),
    code_closed e = true ->
    ∃ r' : renaming,
      renames_alike r r' (f avoid e)
      ∧ rename_conts_alike r r' avoid
      ∧ f (map (rename_continuation r) avoid) (rename_expr r e)
        = rename_expr r' (f avoid e).

(* A rewrite that commutes with renaming, and does nothing with the
   continuations that it is given, does so in that sense: the second
   renaming may be taken to be the first. *)
Definition rewrite_equivariant_up_to_avoiding_statement : Prop :=
  ∀ f : expr -> expr,
    rewrite_equivariant f -> equivariant_up_to_avoiding (fun _ => f).

(* It is kept by a rewrite that commutes with renaming and keeps scope,
   applied before or after: the second renaming is the same one, so it
   still does to the continuations what the first does. *)
Definition equivariant_up_to_avoiding_before_statement : Prop :=
  ∀ (f : list continuation -> expr -> expr) (g : expr -> expr),
    keeps_scope g ->
    rewrite_equivariant g ->
    equivariant_up_to_avoiding f ->
    equivariant_up_to_avoiding (fun avoid e => f avoid (g e)).

Definition equivariant_up_to_avoiding_after_statement : Prop :=
  ∀ (f : list continuation -> expr -> expr) (g : expr -> expr),
    (∀ avoid : list continuation, keeps_scope (f avoid)) ->
    keeps_scope g ->
    rewrite_equivariant g ->
    equivariant_up_to_avoiding f ->
    equivariant_up_to_avoiding (fun avoid e => g (f avoid e)).

(* And by the rewrite applied a number of times with the same
   continuations: after one step the continuations renamed by the
   second renaming are those renamed by the first, so the next step is
   given the same list. *)
Definition equivariant_up_to_avoiding_iterate_statement : Prop :=
  ∀ (rounds : nat) (f : list continuation -> expr -> expr),
    (∀ avoid : list continuation, keeps_scope (f avoid)) ->
    equivariant_up_to_avoiding f ->
    equivariant_up_to_avoiding (fun avoid => iterate rounds (f avoid)).

(* Given no continuation to avoid, such a rewrite commutes with renaming
   up to the names that it makes in the first sense. *)
Definition equivariant_up_to_avoiding_nothing_statement : Prop :=
  ∀ f : list continuation -> expr -> expr,
    equivariant_up_to_avoiding f -> equivariant_up_to (f []).

(* A rewrite [f], then a second one [g], and then a third [h] only
   where [g] changed the term: where the comparison finds what [f]
   gave and what [g] made of it the same up to bound names, the result
   is what [f] gave. A normalisation that has rounds, a rewrite that
   is applied once and the rounds again may be given this form, so
   that the second rounds are run only where the rewrite gave them
   something new. It keeps scope where the three do. And it commutes
   with renaming up to the names that it makes where the three do:
   the second renaming of [f] and that of [g] rename alike the free
   names of what [g] gives, so the comparison answers of the two
   results for the renamed term what it answers of the two results
   for the term, and the same branch is taken. *)
Definition keeps_scope_unless_same_statement : Prop :=
  ∀ f g h : expr -> expr,
    keeps_scope f -> keeps_scope g -> keeps_scope h ->
    keeps_scope
      (fun e => if expr_eqb [] [] (f e) (g (f e)) then f e else h (g (f e))).

Definition equivariant_up_to_unless_same_statement : Prop :=
  ∀ f g h : expr -> expr,
    keeps_scope f -> keeps_scope g -> keeps_scope h ->
    equivariant_up_to f -> equivariant_up_to g -> equivariant_up_to h ->
    equivariant_up_to
      (fun e => if expr_eqb [] [] (f e) (g (f e)) then f e else h (g (f e))).

(* The same where the first and the third are given continuations to
   avoid, the same ones, and the second commutes with renaming and is
   given none. *)
Definition equivariant_up_to_avoiding_unless_same_statement : Prop :=
  ∀ (f h : list continuation -> expr -> expr) (g : expr -> expr),
    (∀ avoid : list continuation, keeps_scope (f avoid)) ->
    keeps_scope g ->
    (∀ avoid : list continuation, keeps_scope (h avoid)) ->
    equivariant_up_to_avoiding f ->
    rewrite_equivariant g ->
    equivariant_up_to_avoiding h ->
    equivariant_up_to_avoiding
      (fun avoid e =>
         if expr_eqb [] [] (f avoid e) (g (f avoid e)) then f avoid e
         else h avoid (g (f avoid e))).

(* ======================================================================== *)
(* The validation, one of whose rewrites makes names                        *)
(* ======================================================================== *)

(* [rename_apart_avoiding] commutes with renaming up to the names that
   it makes in the sense of this file, with the continuations that it
   is given to avoid. Its own file says that the second renaming does
   what the first does to the names that are free in the term and to
   those continuations
   (FlambdaRenameApartSpec.rename_apart_avoiding_equivariant_up_to_statement).
   Here it does so to the names that are free in the result:
   [rename_apart_avoiding] keeps scope
   (FlambdaKeepsScopeSpec.rename_apart_avoiding_keeps_scope_statement).
   So does [rename_apart], which is given nothing to avoid. *)
Definition rename_apart_avoiding_commutes_up_to_statement : Prop :=
  equivariant_up_to_avoiding rename_apart_avoiding.

Definition rename_apart_commutes_up_to_statement : Prop :=
  equivariant_up_to rename_apart.

(* A round of the second phase of the term before keeps scope, and so
   does the normalisation of the term before, which is the rounds of
   the first phase, then those of the second, then the rewrites from
   the types where what they give keeps scope
   (FlambdaGuardScopeSpec.guard_scope of
   FlambdaSimplifyCarriedSpec.simplify_by_types_carried, of the types
   library, which is FlambdaSimplifyPassSpec.simplify_by_types),
   then the rounds of the second phase once more, then the demotion
   of the exception handlers that no call can raise to and the rounds
   that follow it (FlambdaValidateSpec.demote_and_tidy), which
   commutes with renaming and keeps scope: the rewrites that a
   round is made of keep scope (FlambdaKeepsScopeSpec.v,
   FlambdaValidateScopeSpec.v), so does a round of the first phase
   with the step that follows it
   (FlambdaValidateScopeSpec.first_phase_round_keeps_scope_statement),
   and so does a rewrite under that test, whatever the rewrite is
   (FlambdaValidateScopeSpec.guard_scope_keeps_scope_statement).
   The code whose calls are not inlined is named by code ids, which no
   renaming changes. *)
Definition normalise_before_round_keeps_scope_statement : Prop :=
  ∀ kept : list code_id, keeps_scope (normalise_before_round kept).

Definition normalise_before_keeps_scope_statement : Prop :=
  ∀ (rounds : nat) (kept : list code_id),
    keeps_scope (normalise_before rounds kept).

(* The placing of continuations, which the check applies after the
   normalisations, keeps scope. *)
Definition placed_keeps_scope_statement : Prop := keeps_scope placed.

(* So does what the check makes of the term before after that: the
   rewrites of its rounds keep scope (FlambdaKeepsScopeSpec.v,
   FlambdaLoopsScopeSpec.v, and FlambdaStaticsScopeSpec.v for
   FlambdaUnaliasParamsSpec.unalias_params and for
   FlambdaGroupsUpSpec.groups_up, which comes last). *)
Definition tidied_before_keeps_scope_statement : Prop :=
  ∀ rounds : nat, keeps_scope (tidied_before rounds).

(* Before that, when the rounds of the term before are over, the
   check takes its calls on, or leaves the term as it is
   (FlambdaValidateSpec.with_returns): taken or not, the rewrite
   commutes with every renaming and keeps scope. The check chooses by
   whether the term after, as it is compared, has a call that would
   be taken on (FlambdaValidateSpec.keeps_a_return), which is the
   same of a term renamed: the test compares two rewrites of the term
   up to bound names. *)
Definition with_returns_equivariant_statement : Prop :=
  ∀ taken : bool, rewrite_equivariant (with_returns taken).

Definition with_returns_keeps_scope_statement : Prop :=
  ∀ taken : bool, keeps_scope (with_returns taken).

Definition keeps_a_return_equivariant_statement : Prop :=
  ∀ (r : renaming) (e : expr),
    keeps_a_return (rename_expr r e) = keeps_a_return e.

(* A round of the second phase of the term before, and the
   normalisation of the term before, commute with renaming up to the
   names that they make; the first phase commutes with renaming, and
   so do the rewrites from the types under their test of scope, which
   make no name
   (FlambdaValidateScopeSpec.guard_scope_equivariant_statement). These
   stand where this file stated that the normalisation of the term
   before commutes with renaming, which it did while no round made a
   name. *)
Definition normalise_before_round_equivariant_up_to_statement : Prop :=
  ∀ kept : list code_id, equivariant_up_to (normalise_before_round kept).

Definition normalise_before_equivariant_up_to_statement : Prop :=
  ∀ (rounds : nat) (kept : list code_id),
    equivariant_up_to (normalise_before rounds kept).

(* The same of the normalisation of a term in which something was
   lifted, which is given continuations to avoid: after the first
   phase, each of its rounds is a round of the second phase of the
   term before in which the names made avoid those continuations
   (FlambdaValidateSpec.normalise_before_round_avoiding), followed by
   the use of what is known of the static closures in sight, of what
   the unit binds its symbols to, and of the value slots that every
   set of closures of the unit gives one symbol or constant
   (FlambdaKnownSlotsSpec.read_known_slots), which commute with
   renaming and keep scope (FlambdaStaticsScopeSpec.v); between those
   rounds and the same rounds once more come the rewrites from the
   types under their test of scope, as in the normalisation of the
   term before, which commute with renaming, keep scope and make no
   name. The second renaming does to the continuations to avoid what
   the first does. *)
Definition normalise_lifted_keeps_scope_statement : Prop :=
  ∀ (rounds : nat) (avoid : list continuation) (kept : list code_id),
    keeps_scope (normalise_lifted rounds avoid kept).

Definition normalise_lifted_equivariant_up_to_statement : Prop :=
  ∀ (rounds : nat) (kept : list code_id),
    equivariant_up_to_avoiding
      (fun avoid => normalise_lifted rounds avoid kept).

(* The liftings made in turn, each followed by that normalisation
   (FlambdaValidateSpec.lifted_in_turn), keep scope. *)
Definition lifted_in_turn_keeps_scope_statement : Prop :=
  ∀ (rounds : nat) (avoid : list continuation) (kept : list code_id)
    (Ls : list lifting),
    keeps_scope (lifted_in_turn rounds avoid Ls kept).

(* They commute with renaming up to the names that the normalisations
   make, where the continuations to avoid have among them the two of
   the unit of each lifting. The liftings and the continuations to
   avoid are renamed by the first renaming. The term may be under
   another, which does to its free names and to the continuations to
   avoid what the first does: that is how a normalisation leaves the
   term for the lifting after it. The renaming of the result does to
   the free names of the result, and to those continuations, what the
   first does.

   This is not an instance of [equivariant_up_to_avoiding]: the
   liftings are an argument that holds continuations, and the term is
   under a renaming of its own from the start. The proof uses the
   premise on the liftings: a lifting renamed by the first renaming
   is that lifting renamed by the other where the other does to the
   two continuations of its unit what the first does, and the term
   need not mention them. Whether the statement fails without the
   premise is not established. *)
Definition lifted_in_turn_equivariant_up_to_statement : Prop :=
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
        renames_alike r r' (lifted_in_turn rounds avoid Ls kept e)
        ∧ rename_conts_alike r r' avoid
        ∧ lifted_in_turn rounds (map (rename_continuation r) avoid)
            (map (rename_lifting r) Ls) kept (rename_expr r1 e)
          = rename_expr r' (lifted_in_turn rounds avoid Ls kept e).

(* The putting first of the static bindings
   (FlambdaValidateSpec.with_statics_first) makes no name. Its tests go
   by the two continuations of the unit of the first lifting, and the
   check applies it to the term before with the liftings made, which
   is under a renaming of its own by then. So it is stated with the
   liftings under one renaming and the term under another, which does
   to the continuations of the units of the liftings what the first
   does. With one renaming for the two it is an equation with no
   premise. That it keeps scope is in FlambdaStaticsScopeSpec.v. *)
Definition with_statics_first_renamed_alike_statement : Prop :=
  ∀ (r r' : renaming) (Ls : list lifting) (e : expr),
    rename_conts_alike r r' (unit_continuations Ls) ->
    with_statics_first (map (rename_lifting r) Ls) (rename_expr r' e)
    = rename_expr r' (with_statics_first Ls e).

(* The validation answers the same, and for the same reason, on two
   terms and on the two renamed by one renaming, the liftings being
   renamed with them, where the code of the term before is closed. The
   number of rounds, the names of the static bindings to remove, the
   value slots to remove and the code whose calls are not inlined are
   as they were. Nothing is asked of the term after: its normalisation
   commutes with renaming, and so do the removal of unused static
   bindings from its head and the putting of its code first
   ([without_head_statics_equivariant_statement],
   [with_code_first_equivariant_statement]).

   The first comparison is of the two terms as they are, and answers
   the same of the two renamed. After it the direct calls of the code
   of the stubs of the term before are made indirect
   (FlambdaStubCallsSpec.demote_stub_calls), which commutes with the
   renaming, as that renames no code id and keeps what code is
   annotated with, and keeps the code closed; the direct calls of
   code that never returns are put under a binding of their return
   continuation to a handler that is [Invalid]
   (FlambdaNeverReturnsSpec.cut_returns), which commutes with the
   renaming exactly, as it makes no name, and keeps the code closed
   (FlambdaNeverReturnsScopeSpec.v); and the self tail calls
   of the code named are made jumps (FlambdaLoopifySpec.loopify_unit),
   which commutes with the renaming, the names given for the loops
   being renamed with the term ([rename_loopified]), and keeps the
   code closed; what follows is said of the term that this gives. The
   next three are of the term before normalised, which under the
   renaming is that term under a second renaming that does to its
   free names what the first does; for the third of them, and for the
   two comparisons after it, the term after has the unused static
   bindings of its head removed
   (FlambdaValidateSpec.without_head_statics). The last
   two are of the term before with the liftings made in turn, each
   followed by a normalisation
   ([lifted_in_turn_equivariant_up_to_statement]). The first lifting
   commutes with the renaming;
   the normalisation after it makes names, so the term that the next
   lifting is applied to is, under the renaming, that term under
   another renaming, which does to its free names what the first does.
   A lifting holds the two continuations of its unit, and it is given
   renamed by the first renaming. Each normalisation is given the
   continuations of the units of all the liftings to avoid
   (FlambdaValidateSpec.unit_continuations), so the other renaming
   does to them what the first does, and the lifting renamed by the
   first is the lifting renamed by the other, whether or not the term
   still mentions those continuations.
   Each of those terms, and the term after normalised, then has its
   continuations placed and is tidied
   (FlambdaValidateSpec.tidied_before, [tidied_after]), which commutes
   with every renaming and, for the term before, keeps scope
   ([tidied_before_equivariant_statement],
   [tidied_after_equivariant_statement],
   [tidied_before_keeps_scope_statement]).
   The removal of static bindings, the removal of value slots, the
   moving up of the bindings of code, the putting first of the static
   bindings ([with_statics_first_renamed_alike_statement]) and the
   ordering of static bindings commute with those renamings and keep
   scope (FlambdaStaticsScopeSpec.v); and the comparison is up to bound
   names and does not tell apart two renamings that rename the free
   names of its first term alike ([expr_eqb_renamed_alike_statement]).

   What is not said. Nothing is said of a term before with open code.
   Before a round renamed binders apart this statement had no premise.
   While the round after a lifting was not told the continuations of
   the units, the statement had a second premise, on the terms that
   the liftings after the first are applied to; it has no such second
   premise now. *)
Definition validate_unit_equivariant_statement : Prop :=
  ∀ (r : renaming) (rounds : nat) (unused : dropped)
    (slots : list value_slot) (G : loopified) (Ls : list lifting)
    (kept : list code_id) (before after : expr),
    code_closed before = true ->
    validate_unit rounds unused slots (rename_loopified r G)
      (map (rename_lifting r) Ls) kept (rename_expr r before)
      (rename_expr r after)
    = validate_unit rounds unused slots G Ls kept before after.

(* The same where the term before is under a second renaming, which
   does to its free names what the first does, and does so to the
   continuations that the validation is given and the term need not
   mention: those of the units of the liftings and those given for the
   loops. The statement above is the case in which the two renamings
   are one.

   This is the form for a caller that has rewritten the term before,
   ahead of the validation, with a rewrite that makes names: of the
   renamed term such a rewrite gives its result under a second
   renaming. The premise about those continuations is why such a
   rewrite is to be given them to avoid
   ([equivariant_up_to_avoiding]): what
   the second renaming does to the free names of the term says nothing
   of a continuation that the term does not mention, and the
   validation reads these, the lifting for the continuations of its
   unit and FlambdaLoopifySpec.loopify_unit for the continuation of
   each loop. *)
Definition validate_unit_renamed_alike_statement : Prop :=
  ∀ (r r1 : renaming) (rounds : nat) (unused : dropped)
    (slots : list value_slot) (G : loopified) (Ls : list lifting)
    (kept : list code_id) (before after : expr),
    code_closed before = true ->
    renames_alike r r1 before ->
    rename_conts_alike r r1 (unit_continuations Ls ++ map snd G) ->
    validate_unit rounds unused slots (rename_loopified r G)
      (map (rename_lifting r) Ls) kept (rename_expr r1 before)
      (rename_expr r after)
    = validate_unit rounds unused slots G Ls kept before after.

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

(* Which allocations the reaper may unbox, and what the checks of the two
   tables that say so mean.

   This file is to be reviewed: [dominated_by_allocation_point] against
   the last rules of [post_processing_rules] of
   middle_end/flambda2/reaper/points_to_analysis.ml, [unboxing_closed]
   against [unboxing_rules] of middle_end/flambda2/reaper/
   unboxing_analysis.ml and the seeds of [cannot_unbox0] that
   [perform_analysis0] puts in the table before running them, the units
   of the four counterexamples below against their .fl (that of the
   second is [kept_result_unboxed] of FlambdaReaperUnboxing.v),
   [check_result_receivers], [field_reader_seeds] and
   [escaping_argument_seeds] against the fixes they are said to be,
   [unboxing_relations_with] against [unboxing_relations],
   [unbox_dead_unit] against testsuite/tests/reaper/unbox_dead.ml,
   [wrapper_args_u_before_fix] against [make_apply_wrapper] of rebuild.ml
   before the fix that the example is of, and the statements. The checks are
   FlambdaReaperUnboxing.v, which needs no review; the proofs are
   FlambdaReaperUnboxingProofs.v.

   An allocation is unboxed ([to_unbox]) if it has a use and
   [cannot_unbox] does not hold of it; every name whose one allocation
   with a source is that allocation ([dominated_by_allocation_point]) is
   then replaced by variables for the fields read from it. Both tables are
   read off the compiler, and nothing is believed of them:

   - [check_dominated] tests that a table of
     [dominated_by_allocation_point] is that relation, both ways
     ([check_dominated_sound_statement]): a name missing from it would
     keep a reference to an allocation that is no longer made, and a name
     wrongly in it would be replaced by the fields of an allocation it
     may not be.
   - [check_to_unbox] tests that a table of [to_unbox] is the relation
     that the rules give, both ways, for tables of the other relations of
     [unboxing_rules] that are closed under the rules
     ([check_to_unbox_sound_statement]): an allocation is in [to_unbox]
     if and only if it has a use and is not in [cannot_unbox], leaving
     out allocations with no source, which dominate no name. Both ways
     are needed: an allocation that a rule forbids to unbox is unboxed
     wrongly, and one that is left out may hold in a field an allocation
     that is unboxed on the assumption that it is unboxed too (rule
     [cu0_stored]), whose field would then be poison. The statement does
     not say that the closed tables are the least: tables that are larger
     make [cannot_unbox] larger and [to_unbox] smaller, consistently, and
     what the rebuild is to rely on is that agreement. The rules are
     monotone in the relations they derive (their negated premises are of
     the tables of points_to_analysis.ml and of the flags), so the least
     tables, which the compiler computes, are in any closed ones; the
     check computes them too, [cannot_unbox0] by rounds of its one
     recursive rule, and tests that they are closed, so that the table of
     the compiler passes.
   - [check_to_change_representation] tests the same of a table of
     [to_change_representation], the last rule, for a table of
     [to_unbox] that passes its check
     ([check_to_change_representation_sound_statement]): an allocation
     changes its representation if and only if it has a use, is not in
     [cannot_change_representation] and is not unboxed, leaving out
     allocations with no source. Both ways are needed: a closure that
     changes its representation wrongly may be read by code that expects
     its old slots, and one that is left out keeps in a slot a block
     unboxed on the assumption that the closure would hold its fields
     (rule [cu0_stored]).
   - [check_sets_whole] tests that the closures of a set that have a use
     are unboxed alike and change their representation alike
     ([check_sets_whole_sound_statement]): the rebuild unboxes a set, or
     changes its representation, whole.

   [unboxing_closed] says that five tables, those of
   [cannot_change_representation0], [cannot_change_representation1],
   [cannot_change_representation], [cannot_unbox0] and [cannot_unbox],
   are closed under the rules, each a field named after the relation it
   concludes and the rule; the rules of [to_unbox] and
   [to_change_representation] are the statements of their checks.
   Nothing is said of a table that fails a check; the examples of
   FlambdaReaperUnboxing.v have tables that the checks reject, and tables
   that pass.

   A counterexample, found by fuzzing: a crash of the compiler's reaper
   with lambda lifting, which the compiler has fixed, and the model with
   it (middle_end/flambda2/tests/semantics/reaper/
   lambda_lifting_dead_callee.fl, whose comment gives the OCaml). With
   lambda lifting, a closure that is only called may be unboxed, and so
   is the [my_closure] of its code if the closure dominates it: a call
   then gives the fields of the closure for those of [my_closure], and
   has no callee ([Unbox_my_closure]). Two closures of one code, as two
   inlined copies of a function make them, are each unboxed, but
   [my_closure], of which they are the two sources, is dominated by
   neither and is not unboxed. The decision on [my_closure] is then
   [Keep_my_closure], and the rebuild kept the callee of each call: a
   closure whose binding the unboxing removed. The compiler's fix, in
   the commit "reaper: fix a few bugs", makes [rewrite_simple_opt] of
   rebuild.ml drop a callee with unboxed fields, so that the call has no
   callee under [Keep_my_closure] too; the model writes it poison. (The
   other fix of the fuzzing report, not to unbox a called closure unless
   the [my_closure] of its code is unboxed too, is no longer needed, and
   its check is gone.)
   [lambda_lifting_dead_callee_statement] says that the tables of such a
   unit pass every check, with lambda lifting and without, that the unit
   exits, and that rebuilt it exits as the unit does: without lambda
   lifting as it is, and with lambda lifting once its calls are given a
   closure of their code for callee. The machine runs a call by the
   closure that its callee is (FlambdaSemanticsSpec.shape_of_closure_call),
   so that a call with no callee is undefined there; the code does not
   read its closure, and the compiler compiles such a call.
   [lambda_lifting_dead_callee_cause_statement] says why the callees are
   dropped.

   A second counterexample, a miscompile of the compiler's reaper that
   the model shares: [kept_result_unboxed] of FlambdaReaperUnboxing.v
   (middle_end/flambda2/tests/semantics/reaper/unbox_kept_result.fl,
   whose comment gives the OCaml). The result of code is unboxed if one
   allocation dominates it and has a use, which a call of the code whose
   continuation reads a field of the result gives it. Another call may
   return to a continuation whose parameter has another source, a
   closure, and is only called: the parameter is dominated by no
   allocation, and is kept. [make_apply_wrapper] of rebuild.ml then has
   the decision [Keep] of the continuation for a result that the callee
   unboxes, and makes the call one that never returns, the test that
   would stop with a fatal error being stubbed ("CR ncourant: restore
   this check"). [kept_result_unboxed_statement] says that the tables of
   the unit pass every check, that the unit exits, and that rebuilt it
   goes wrong; [kept_result_unboxed_cause_statement] says why.

   The fix: the analysis must not unbox an allocation that code returns
   unless, at each call of the code, the name that the call gives it to,
   the parameter of the continuation, is unboxed too.
   [check_result_receivers] is what the fix asks of a table of
   [to_unbox]. As a rule of [unboxing_rules], the fix is a seed of
   [cannot_unbox0] like [cu0_result], the rule of unboxing_analysis.ml
   that begins "An allocation that is one of the results of a function",
   with the premises
     [usages call_witness usage;
      rev_accessor ~base:usage relation ~to_:v;
      ~~(dominated_by_allocation_point allocation_id v)]
   in place of
     [constructor ~base:call_witness !!Field.code_id_of_call_witness
        ~from:codeid;
      cannot_change_calling_convention codeid]:
   the first two find [v], a name that a call reads the result into, as
   in the rule of [field_of_constructor_is_used] of
   [post_processing_rules] that has them; the third, a negated premise on
   a table of points_to_analysis.ml as the others of the rules are, says
   that [v] is not unboxed with the allocation.
   [kept_result_unboxed_statement] says too that the check rejects the
   tables of the unit, and passes those of [unboxed_closure] and of the
   four examples of blocks of FlambdaReaperUnboxing.v. In the unit, the
   rule seeds [cannot_unbox0] with the block [x42] alone, as
   [cu0_max_unbox_size] does with a size of 0 ([x42] has the one used
   real field of the unit): the analysis with the fix gives the tables
   with that size, those of the unit with [x42] left out of [to_unbox],
   which pass every check and that of the fix, and whose rebuild exits as
   the unit does.

   A miscompile of the compiler's reaper that the compiler has fixed, and
   the model with it, by "reaper: fix miscompilation with unused unboxed
   variables" (#7952): [unbox_dead_unit], the unit of the compiler's test
   testsuite/tests/reaper/unbox_dead.ml with a block for the boxed float.
   The result of code is unboxed, as a call of the code reads a field of
   it. Another call returns to a continuation that has another use, from
   the other branch of a conditional, with a constant: its parameter has
   two sources and is not unboxed, and has no use, and is deleted.
   [make_apply_wrapper] then has the decision [Delete] of the
   continuation for a result that the callee unboxes; before the fix, it
   took that for a disagreement on unboxing and made the call one that
   never returns, and it now drops the result. [unbox_dead_statement]
   says that the tables of the unit pass every check, that the unit
   exits, and that rebuilt it exits as the unit does; and that the match
   of [wrapper_args_u] in its order before the fix
   ([wrapper_args_u_before_fix]) gives no arguments for that call, whose
   wrapper is then [Invalid], with which the rebuilt unit goes wrong.

   A third counterexample, on which the compiler's reaper stops and the
   model's miscompiles: [mixed_field_unit]
   (middle_end/flambda2/tests/semantics/reaper/mixed_mutable_field.fl,
   whose comment gives the OCaml). A field is named by its index and its
   kind, not its mutability: under a GADT, a field read immutably at one
   type and mutably at another is one field. The pair that the field
   holds dominates the immutable read, and is unboxed; the mutable read
   has any source (the traversal makes the result of a mutable load so),
   and is kept. The compiler stops in [mk_unboxed_fields] of
   unboxing_analysis.ml ("flows to both unboxed and non-unboxed
   variables"). The model does not: it unboxes the block that holds the
   pair, and the parameter of the code that it dominates, to that field,
   a value, which no longer exists; its rebuild makes the call, and the
   arm that reads the pair, [Invalid]. [mixed_mutable_field_statement]
   says that the tables of the unit pass every check, that the unit
   exits with 3, and that rebuilt it goes wrong;
   [mixed_mutable_field_cause_statement] says why.

   The fix: the analysis must not unbox an allocation stored in a field
   that a use of the allocation that holds it reads into a name with a
   use that the allocation does not dominate. As a rule of
   [unboxing_rules], it is a seed of [cannot_unbox0]
   ([field_reader_seeds]), with the premises
     [sources alias allocation_id; has_source allocation_id;
      constructor ~base relation ~from:alias; usages base usage;
      rev_accessor ~base:usage relation ~to_:v; has_usage v;
      ~~(dominated_by_allocation_point allocation_id v)]:
   those of the rule of [field_of_constructor_is_used] of
   [post_processing_rules] that has [usages], for a source of the field,
   and the negated premise of the fix of [kept_result_unboxed].
   [check_field_readers], what it asks of a table of [to_unbox], is
   [check_result_receivers] for every field, not only the results of
   calls, and for the names with a use only: [check_result_receivers]
   rejects the tables of [unbox_dead_unit], whose [x32] has no use. A
   field named by its mutability too, the other way, is a change of the
   traversal that the model does not make. The statement says too that
   the check rejects the tables of the unit and passes those of the
   sound examples, and that with the seeds of the fix
   ([unboxing_relations_with]) the tables are closed under the rules,
   pass the check, and rebuild into a unit that exits with 3.

   A fourth counterexample, a crash of the compiler's reaper that the
   model shares: [escaping_argument_unit]
   (middle_end/flambda2/tests/semantics/reaper/
   escaping_parameter_argument.fl, whose comment gives the OCaml). Code
   that cannot change its calling convention, as it escapes, has a
   parameter of any source, which it only calls (in an arm that a GADT
   makes dead); a call of the code gives it a block, which nothing else
   uses. The rule that forbids to unbox it, [cu0_parameter] (with its CR
   in unboxing_analysis.ml), reads the sources of the parameter, and
   those of a name of any source are not recorded: the block is unboxed.
   The case [Not_changing_calling_convention] of [rebuild_apply]
   (rebuild.ml) passes the argument, whose decision is [Keep], as it is:
   the compiler stops ("Incorrect free names"), and the model's rebuild
   gives the call the variable of the block, whose binding it removed,
   and goes wrong. [escaping_parameter_argument_statement] says that the
   tables of the unit pass every check, that the unit exits, and that
   rebuilt it goes wrong; [escaping_parameter_argument_cause_statement]
   says why.

   The fix has two halves. The first: the analysis must not unbox an
   allocation that is a source of an argument of a call of code that
   cannot change its calling convention, whatever the sources of the
   parameter. As a rule of [unboxing_rules], it is a seed of
   [cannot_unbox0] ([escaping_argument_seeds]) like [cu0_parameter], with
     [sources arg allocation_id; has_source allocation_id;
      argument ~from:arg relation ~base:apply;
      sources apply call_witness]
   in place of
     [sources alias allocation_id; has_source allocation_id;
      rev_parameter ~to_:alias relation ~base:call_witness]:
   the argument, whose sources are recorded, and the witnesses that the
   call may call, for the parameter. [check_escaping_arguments] is what
   it asks of a table of [to_unbox]. The statement says that the check
   rejects the tables of the unit and passes those of the sound
   examples, and that with the seeds of the fix the tables are closed
   under the rules, pass the check, and rebuild into a unit that exits as
   the unit does. The second half, in rebuild.ml: [rewrite_simple] of
   the arguments that the case [Not_changing_calling_convention] keeps,
   which makes the unboxed one poison. The statement says that the
   rebuilt unit is the unit without the block, whose variable it gives
   the call, and that with poison for that argument it exits as the unit
   does. The second half alone is sound only where the code uses the
   parameter on no path that is taken: here, as the parameter is only
   called in an arm that is not. Of the uses of a name of any source, the
   rules let only a call leave the allocation unboxed (a real field read
   forbids it by [crr0_any_source_usage], an escape by [cu0_any_usage]),
   and a call of a block goes wrong in the input too; that a closure so
   called is never unboxed is not proved. The first half is a check of
   the tables, and needs none of this. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaPrimitiveModel FlambdaMapSpec FlambdaBindersSpec FlambdaValueSpec
  FlambdaSemanticsSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperRebuild FlambdaReaperCallingConventions FlambdaReaperGraphSpec
  FlambdaReaperUnboxing.
From OxCamlFlambda2 Require FlambdaReaperTupledSpec.
Import ListNotations.

(* ======================================================================== *)
(* Allocation points                                                        *)
(* ======================================================================== *)

Section Allocation_points.

Variable S : solution.

(* [multiple_allocation_points x]: [x] may come from any allocation, or
   from two different ones that have a source. *)
Definition multiple_allocation_points_of (x : node) : Prop :=
  s_any_source S x = true
  ∨ ∃ y z : node,
      s_sources S x y = true ∧ s_has_source S y = true
      ∧ s_sources S x z = true ∧ s_has_source S z = true ∧ y <> z.

(* [dominated_by_allocation_point a x]: [a] is the one allocation with a
   source that [x] may come from. *)
Definition dominated_by_allocation_point (a x : node) : Prop :=
  s_sources S x a = true ∧ s_has_source S a = true
  ∧ ¬ multiple_allocation_points_of x.

End Allocation_points.

Definition check_dominated_sound_statement : Prop :=
  ∀ (T : tables) (dominated : list (node * node)),
    check_dominated T dominated = true ->
    ∀ a x : node,
      mem_pair a x dominated = true
      <-> dominated_by_allocation_point (solution_of_tables T) a x.

(* ======================================================================== *)
(* The rules of the unboxing                                                *)
(* ======================================================================== *)

Section Rules.

Variable is_local : field -> bool.
Variable G : graph.
Variable S : solution.
(* [cannot_change_calling_convention]. *)
Variable cannot_change : node -> bool.
(* [Flambda_features.reaper_max_unbox_size] and the extra option
   [reaper-lambda-lifting]. *)
Variable max_unbox_size : nat.
Variable lambda_lifting : bool.
(* [cannot_change_representation0], [cannot_change_representation1],
   [cannot_change_representation], [cannot_unbox0], [cannot_unbox]. *)
Variables crr0 crr1 crr cu0 cu : node -> bool.

Abbreviation any_usage := (s_any_usage S).
Abbreviation any_source := (s_any_source S).
Abbreviation has_usage := (s_has_usage S).
Abbreviation has_source := (s_has_source S).
Abbreviation usages := (s_usages S).
Abbreviation sources := (s_sources S).

(* The rules of [unboxing_rules], in their order there, and the seeds of
   [cannot_unbox0] of [perform_analysis0] ([cu0_max_unbox_size]), each a
   field named after the relation it concludes and the rule; each says
   that the tables have the conclusion of the rule wherever they have the
   premises. *)
Record unboxing_closed : Prop := {
  (* An allocation that escapes with a real field that is not local. *)
  crr0_any_usage : ∀ x field y,
    any_usage x = true -> is_local field = false ->
    is_real_field field = true -> g_constructor G x field y ->
    crr0 x = true;
  (* An allocation that escapes with a local field that is read from an
     allocation of any source. *)
  crr0_local_reading : ∀ x field y z,
    any_usage x = true -> is_local field = true ->
    s_reading_field S field z = true -> g_constructor G x field y ->
    crr0 x = true;
  (* A local field read from a value of two sources. *)
  crr0_local_sources : ∀ usage field v source1 source2,
    g_accessor G v field usage -> has_usage v = true ->
    is_local field = true ->
    sources usage source1 = true -> has_source source1 = true ->
    sources usage source2 = true -> has_source source2 = true ->
    source1 <> source2 ->
    crr0 source1 = true;
  (* A use that reads a real field of a value of another source too. *)
  crr0_other_source : ∀ allocation_id alias alias_source field v,
    usages allocation_id alias = true ->
    sources alias alias_source = true -> has_source alias_source = true ->
    alias_source <> allocation_id -> is_real_field field = true ->
    g_accessor G v field alias -> has_usage v = true ->
    crr0 allocation_id = true;
  (* A use that reads a real field of a value of any source. *)
  crr0_any_source_usage : ∀ allocation_id alias field v,
    usages allocation_id alias = true -> any_source alias = true ->
    is_real_field field = true -> g_accessor G v field alias ->
    has_usage v = true ->
    crr0 allocation_id = true;
  crr0_other_own_source : ∀ allocation_id source,
    sources allocation_id source = true -> has_source source = true ->
    source <> allocation_id ->
    crr0 allocation_id = true;
  crr0_not_own_source : ∀ allocation_id usage,
    usages allocation_id usage = true -> has_usage usage = true ->
    sources allocation_id allocation_id = false ->
    crr0 allocation_id = true;
  crr0_any_source : ∀ allocation_id,
    any_source allocation_id = true -> crr0 allocation_id = true;
  crr0_call_witness : ∀ call_witness code_id,
    g_constructor G call_witness Field_code_id_of_call_witness code_id ->
    crr0 call_witness = true;
  crr1_crr0 : ∀ x, crr0 x = true -> crr1 x = true;
  (* The closures of a set go together. *)
  crr1_function_slot : ∀ x field y,
    g_constructor G x field y -> is_function_slot field = true ->
    crr0 x = true -> crr1 y = true;
  crr_crr1 : ∀ x, crr1 x = true -> crr x = true;
  (* Blocks and boxed numbers never change their representation. *)
  crr_block_like : ∀ x field y,
    g_constructor G x field y -> is_block_like_field field = true ->
    crr x = true;
  cu0_cannot_change_representation : ∀ x, crr1 x = true -> cu0 x = true;
  cu0_any_usage : ∀ x, any_usage x = true -> cu0 x = true;
  (* A closure of code that cannot change its calling convention. *)
  cu0_called_cannot_change : ∀ x call_witness codeid,
    g_constructor G x (Field_call_witness Known_arity_code_pointer)
      call_witness ->
    g_constructor G call_witness Field_code_id_of_call_witness codeid ->
    cannot_change codeid = true ->
    cu0 x = true;
  (* A closure that is called, unless closures are lambda-lifted. *)
  cu0_called : ∀ x call_witness,
    g_constructor G x (Field_call_witness Known_arity_code_pointer)
      call_witness ->
    has_usage call_witness = true -> lambda_lifting = false ->
    cu0 x = true;
  (* A result of code that cannot change its calling convention. *)
  cu0_result : ∀ alias allocation_id relation call_witness codeid,
    sources alias allocation_id = true -> has_source allocation_id = true ->
    g_constructor G call_witness relation alias ->
    is_return_of_call relation = true ->
    g_constructor G call_witness Field_code_id_of_call_witness codeid ->
    cannot_change codeid = true ->
    cu0 allocation_id = true;
  (* A parameter of code that cannot change its calling convention. *)
  cu0_parameter : ∀ alias allocation_id relation call_witness codeid,
    sources alias allocation_id = true -> has_source allocation_id = true ->
    g_parameter G call_witness relation alias ->
    g_constructor G call_witness Field_code_id_of_call_witness codeid ->
    cannot_change codeid = true ->
    cu0 allocation_id = true;
  cu0_symbol : ∀ x source,
    sources x source = true -> is_symbol_node x = true -> cu0 x = true;
  (* An allocation stored in a used real field of one that can neither
     change its representation nor be unboxed. *)
  cu0_stored : ∀ alias relation to_ allocation_id,
    sources alias allocation_id = true ->
    g_constructor G to_ relation alias ->
    s_field_used S to_ relation = true -> crr to_ = true ->
    is_real_field relation = true -> cu0 to_ = true ->
    cu0 allocation_id = true;
  (* An allocation with more used real fields, function slots apart, than
     [max_unbox_size]. *)
  cu0_max_unbox_size : ∀ (x : node) (fields : list field),
    NoDup fields ->
    (∀ f : field,
       In f fields ->
       s_field_used S x f = true ∧ is_real_field f = true
       ∧ is_function_slot f = false) ->
    max_unbox_size < length fields ->
    cu0 x = true;
  cu_cu0 : ∀ x, cu0 x = true -> cu x = true;
  cu_function_slot : ∀ x field y,
    cu0 x = true -> g_constructor G x field y ->
    is_function_slot field = true ->
    cu y = true
}.

End Rules.

(* ======================================================================== *)
(* The check                                                                *)
(* ======================================================================== *)

Definition check_to_unbox_sound_statement : Prop :=
  ∀ (is_local : field -> bool) (F : facts) (T : tables)
    (cannot_change : list node) (max_unbox_size : nat)
    (lambda_lifting : bool) (to_unbox : list node),
    check_to_unbox is_local F T cannot_change max_unbox_size lambda_lifting
      to_unbox = true ->
    ∃ crr0 crr1 crr cu0 cu : node -> bool,
      unboxing_closed is_local (graph_of_facts F) (solution_of_tables T)
        (fun n => mem_node n cannot_change) max_unbox_size lambda_lifting
        crr0 crr1 crr cu0 cu
      ∧ (∀ x : node,
           mem_node x to_unbox = true ->
           s_has_usage (solution_of_tables T) x = true ∧ cu x = false)
      ∧ (∀ x : node,
           s_has_usage (solution_of_tables T) x = true ->
           s_has_source (solution_of_tables T) x = true -> cu x = false ->
           mem_node x to_unbox = true).

Definition check_to_change_representation_sound_statement : Prop :=
  ∀ (is_local : field -> bool) (F : facts) (T : tables)
    (cannot_change : list node) (max_unbox_size : nat)
    (lambda_lifting : bool) (to_unbox tcr : list node),
    check_to_change_representation is_local F T cannot_change max_unbox_size
      lambda_lifting to_unbox tcr = true ->
    ∃ crr0 crr1 crr cu0 cu : node -> bool,
      unboxing_closed is_local (graph_of_facts F) (solution_of_tables T)
        (fun n => mem_node n cannot_change) max_unbox_size lambda_lifting
        crr0 crr1 crr cu0 cu
      ∧ (∀ x : node,
           mem_node x to_unbox = true ->
           s_has_usage (solution_of_tables T) x = true ∧ cu x = false)
      ∧ (∀ x : node,
           s_has_usage (solution_of_tables T) x = true ->
           s_has_source (solution_of_tables T) x = true -> cu x = false ->
           mem_node x to_unbox = true)
      ∧ (∀ x : node,
           mem_node x tcr = true ->
           s_has_usage (solution_of_tables T) x = true ∧ crr x = false
           ∧ mem_node x to_unbox = false)
      ∧ (∀ x : node,
           s_has_usage (solution_of_tables T) x = true ->
           s_has_source (solution_of_tables T) x = true -> crr x = false ->
           mem_node x to_unbox = false ->
           mem_node x tcr = true).

(* ======================================================================== *)
(* Sets of closures                                                         *)
(* ======================================================================== *)

(* Two closures of a set, one held by the other in a function slot, that
   have a use are both unboxed or neither, and both change their
   representation or neither. *)
Definition check_sets_whole_sound_statement : Prop :=
  ∀ (F : facts) (T : tables) (to_unbox tcr : list node),
    check_sets_whole F T to_unbox tcr = true ->
    ∀ (x y : node) (f : field),
      g_constructor (graph_of_facts F) x f y -> is_function_slot f = true ->
      s_has_usage (solution_of_tables T) x = true ->
      s_has_usage (solution_of_tables T) y = true ->
      mem_node x to_unbox = mem_node y to_unbox
      ∧ mem_node x tcr = mem_node y tcr.

(* ======================================================================== *)
(* A counterexample: lambda lifting and a dead callee                       *)
(* ======================================================================== *)

Import Examples.
Local Open Scope positive_scope.

(* The code 54 returns its parameter, and does not use its closure. The
   unit makes two closures of it, [x60] and [x61], as two inlined copies
   of a function would; it calls the code through [x60] with 9, then
   through [x61] with 9, and returns the result. *)
Definition dead_callee_code : code :=
  Code [40] 41 3 4 (make_code_metadata [Singleton Value] [Singleton Value])
    (return_to 3 [Var 40]).

Definition dead_callee_unit : expr :=
  Let_static [Bind_code 54 dead_callee_code]
    (Let 60 Value (Closure_of_set 70 [(70, 54)] [])
       (Let 61 Value (Closure_of_set 70 [(70, 54)] [])
          (Let_cont_non_recursive 20
             (Handler [30] (make_handler_metadata [Value])
                (Apply (call 54 (Var 61))))
             (Apply (with_continuation (call 54 (Var 60)) 20))))).

Definition dead_callee_facts : facts :=
  traverse_unit Auto interface dead_callee_unit.

Definition dead_callee_code_deps : finite_map code_dep :=
  unit_code_deps Auto interface dead_callee_unit.

(* The solution of the unit, as the solver of the test tool computes it.
   The code dep of 54 has the result 3, the exception 4, [my_closure] 5,
   the parameter 6 and the witnesses 7 (of known arity) and 8; the calls
   make the nodes 9 to 18. [my_closure] has the two sources [x60] and
   [x61]. *)
Definition dead_callee_tables : tables :=
  {| t_alias := [(m 18, m 7); (m 5, m 17); (m 13, m 7); (m 5, m 12);
       (m 14, m 16); (m 9, m 11); (m 3, v 40); (v 41, m 5); (v 40, m 6);
       (m 16, m 7); (m 11, m 7); (m 1, m 3); (v 30, m 3); (m 2, m 4);
       (m 10, c 54); (m 15, c 54); (m 6, consts); (m 5, v 60);
       (m 5, v 61)];
     t_any_usage := [m 1; m 15; m 10; m 2; m 3; m 4; c 54; v 40; m 6;
       consts];
     t_any_source := [c 54; consts; m 10; m 15; m 6; v 40; m 3; m 1; v 30];
     t_has_usage := [m 1; m 15; m 10; m 2; m 14; m 9; m 16; m 11; m 7; m 3;
       m 4; c 54; v 40; m 6; consts; v 60; v 61];
     t_has_source := [c 54; consts; v 61; v 60; m 8; m 7; m 18; m 13; m 16;
       m 11; m 10; m 15; m 6; m 14; m 9; v 40; m 5; m 3; v 41; m 1; v 30];
     t_usages := [(v 60, v 60); (m 14, m 14); (v 61, v 61); (m 9, m 9);
       (m 16, m 14); (m 11, m 9); (m 7, m 14); (m 7, m 9)];
     t_sources := [(v 60, v 60); (v 61, v 61); (m 8, m 8); (m 7, m 7);
       (m 18, m 7); (m 13, m 7); (m 16, m 7); (m 11, m 7); (m 14, m 7);
       (m 9, m 7); (m 5, v 60); (m 5, v 61); (v 41, v 61); (v 41, v 60)];
     t_reading_field := [];
     t_escaping_field := [];
     t_field_used := [(v 60, known); (v 61, known); (m 7, result 0);
       (m 7, Field_return_of_call Exn);
       (m 7, Field_code_id_of_call_witness)] |}.

Definition dead_callee_cannot_change : list node :=
  derive_cannot_change dead_callee_facts dead_callee_tables.

Definition dead_callee_dominated : list (node * node) :=
  derive_dominated dead_callee_tables.

(* The tables of [to_unbox] and [to_change_representation] that the rules
   give, with lambda lifting or without, and the unit rebuilt with them. *)
Definition dead_callee_to_unbox (lambda_lifting : bool) : list node :=
  derive_to_unbox no_local dead_callee_facts dead_callee_tables
    dead_callee_cannot_change max_unbox_size lambda_lifting.

Definition dead_callee_tcr (lambda_lifting : bool) : list node :=
  derive_to_change_representation no_local dead_callee_facts
    dead_callee_tables dead_callee_cannot_change max_unbox_size
    lambda_lifting (dead_callee_to_unbox lambda_lifting).

Definition dead_callee_rebuilt (lambda_lifting : bool) : expr :=
  rebuild_unit_unbox_repr no_local dead_callee_facts dead_callee_tables
    dead_callee_cannot_change dead_callee_code_deps dead_callee_dominated
    (dead_callee_to_unbox lambda_lifting) (dead_callee_tcr lambda_lifting)
    true dead_callee_unit.

Definition dead_callee_exit : behaviour :=
  Exits 1 [Value_const (Tagged_immediate 9)] empty_heap.

(* The calls of the unit rebuilt with lambda lifting, through [callee]:
   the first returns to 62, which drops its result, the second from the
   unit. *)
Definition dead_callee_call (callee : simple) (k : continuation) : expr :=
  Apply
    {| ap_callee := callee; ap_call_kind := Direct 54; ap_args := [int 9];
       ap_continuation := k; ap_exn_continuation := 2;
       ap_exn_extra_args := [];
       ap_args_arity := [Unboxed_product [Singleton Value]];
       ap_return_arity := [Unboxed_product [Singleton Value]] |}.

Definition dead_callee_calls (callee : simple) : expr :=
  Let_cont_non_recursive 20
    (Handler [] (make_handler_metadata []) (dead_callee_call callee 1))
    (Let_cont_non_recursive 62
       (Handler [62] (make_handler_metadata [Value]) (return_to 20 []))
       (dead_callee_call callee 62)).

(* With lambda lifting and without, the tables pass every check. The unit
   exits with 9, and so does it rebuilt: without lambda lifting; with
   lambda lifting, where it is the code 54 and calls with no callee
   (poison), once a closure of 54 is given to the calls for callee. *)
Definition lambda_lifting_dead_callee_statement : Prop :=
  (∀ lambda_lifting : bool,
     let to_unbox := dead_callee_to_unbox lambda_lifting in
     let tcr := dead_callee_tcr lambda_lifting in
     check_closed no_local dead_callee_facts dead_callee_tables = true
     ∧ check_field_used_exact no_local dead_callee_facts dead_callee_tables
       = true
     ∧ check_cc_closed dead_callee_facts dead_callee_tables
         dead_callee_cannot_change = true
     ∧ check_dominated dead_callee_tables dead_callee_dominated = true
     ∧ check_to_unbox no_local dead_callee_facts dead_callee_tables
         dead_callee_cannot_change max_unbox_size lambda_lifting to_unbox
       = true
     ∧ check_to_change_representation no_local dead_callee_facts
         dead_callee_tables dead_callee_cannot_change max_unbox_size
         lambda_lifting to_unbox tcr = true
     ∧ check_sets_whole dead_callee_facts dead_callee_tables to_unbox tcr
       = true)
  ∧ behaves (FlambdaReaperTupledSpec.on_its_own dead_callee_unit)
      dead_callee_exit
  ∧ behaves (FlambdaReaperTupledSpec.on_its_own (dead_callee_rebuilt false))
      dead_callee_exit
  ∧ ∃ c : code,
      dead_callee_rebuilt true
      = Let_static [Bind_code 54 c] (dead_callee_calls (Const (Poison Value)))
      ∧ behaves
          (FlambdaReaperTupledSpec.on_its_own
             (Let_static [Bind_code 54 c]
                (Let 60 Value (Closure_of_set 70 [(70, 54)] [])
                   (dead_callee_calls (Var 60)))))
          dead_callee_exit.

(* Why. With lambda lifting, the two closures are unboxed, to no field;
   [my_closure] has two allocation points, and is not: the decision on it
   is [Keep_my_closure] ([None]). The rebuild drops the callee of each
   call, which has unboxed fields: the rebuilt unit has no free variable
   (before the fix, it had [x61] and [x60]). Its run, which needs a
   closure for each call, goes wrong at the first. Without lambda
   lifting, nothing is unboxed. *)
Definition lambda_lifting_dead_callee_cause_statement : Prop :=
  dead_callee_to_unbox true = [v 60; v 61]
  ∧ fst (unit_unboxed_fields dead_callee_facts dead_callee_tables
           dead_callee_dominated (dead_callee_to_unbox true)
           dead_callee_unit)
    = [(v 60, []); (v 61, [])]
  ∧ multiple_allocation_points dead_callee_tables (m 5) = true
  ∧ unit_my_closure_decisions_u dead_callee_facts dead_callee_tables
      dead_callee_cannot_change dead_callee_code_deps dead_callee_dominated
      (dead_callee_to_unbox true) dead_callee_unit
    = [(54, None)]
  ∧ free_variables_of_expr dead_callee_unit = []
  ∧ free_variables_of_expr (dead_callee_rebuilt true) = []
  ∧ behaves (FlambdaReaperTupledSpec.on_its_own (dead_callee_rebuilt true))
      Goes_wrong
  ∧ dead_callee_to_unbox false = [].

(* ======================================================================== *)
(* A counterexample: a result kept by a continuation and unboxed by code    *)
(* ======================================================================== *)

(* The fix, as what it asks of a table of [to_unbox]: an allocation to
   unbox with a source that is a source of a result of code, a field
   [Return_of_call] of a witness of the code (as in [cu0_result]),
   dominates each name that a call of the code, a use of the witness,
   reads that field into. *)
Definition check_result_receivers (F : facts) (T : tables)
    (dominated : list (node * node)) (to_unbox : list node) : bool :=
  forallb
    (fun '(call_witness, relation, alias) =>
       negb (is_return_of_call relation)
       || forallb
            (fun '(alias', allocation_id) =>
               negb (node_eqb alias alias')
               || negb (mem_node allocation_id (t_has_source T))
               || negb (mem_node allocation_id to_unbox)
               || forallb
                    (fun '(v, relation', usage) =>
                       negb (field_eqb relation relation')
                       || negb (mem_pair call_witness usage (t_usages T))
                       || mem_pair allocation_id v dominated)
                    (f_accessor F))
            (t_sources T))
    (f_constructor F).

(* The unit [kept_result_unboxed] and its tables are those of
   FlambdaReaperUnboxing.v. *)
Definition kept_result_cannot_change : list node :=
  derive_cannot_change kept_result_unboxed_facts kept_result_unboxed_tables.

Definition kept_result_code_deps : finite_map code_dep :=
  unit_code_deps Auto interface kept_result_unboxed.

(* The size of an unboxing without the fix, and with it: the fix seeds
   [cannot_unbox0] with [x42] alone, as [cu0_max_unbox_size] does with a
   size of 0. *)
Definition kept_result_size (fixed : bool) : nat :=
  if fixed then 0 else max_unbox_size.

(* The tables of [to_unbox] and [to_change_representation] that the rules
   give, without the fix and with it, and the unit rebuilt with them. *)
Definition kept_result_to_unbox (fixed : bool) : list node :=
  derive_to_unbox no_local kept_result_unboxed_facts
    kept_result_unboxed_tables kept_result_cannot_change
    (kept_result_size fixed) false.

Definition kept_result_tcr (fixed : bool) : list node :=
  derive_to_change_representation no_local kept_result_unboxed_facts
    kept_result_unboxed_tables kept_result_cannot_change
    (kept_result_size fixed) false (kept_result_to_unbox fixed).

Definition kept_result_rebuilt (fixed : bool) : expr :=
  rebuild_unit_unbox_repr no_local kept_result_unboxed_facts
    kept_result_unboxed_tables kept_result_cannot_change
    kept_result_code_deps kept_result_unboxed_dominated
    (kept_result_to_unbox fixed) (kept_result_tcr fixed) true
    kept_result_unboxed.

Definition kept_result_exit : behaviour :=
  Exits 1 [Value_const (Tagged_immediate 10)] empty_heap.

(* Without the fix and with it, the tables pass every check. The unit
   exits with 10; rebuilt without the fix, it goes wrong, and with it, it
   exits with 10. The check of the fix rejects the tables without it, and
   passes those with it, and those of [unboxed_closure] and of the four
   examples of blocks. *)
Definition kept_result_unboxed_statement : Prop :=
  (∀ fixed : bool,
     let size := kept_result_size fixed in
     let to_unbox := kept_result_to_unbox fixed in
     let tcr := kept_result_tcr fixed in
     check_closed no_local kept_result_unboxed_facts
       kept_result_unboxed_tables = true
     ∧ check_field_used_exact no_local kept_result_unboxed_facts
         kept_result_unboxed_tables = true
     ∧ check_cc_closed kept_result_unboxed_facts kept_result_unboxed_tables
         kept_result_cannot_change = true
     ∧ check_dominated kept_result_unboxed_tables
         kept_result_unboxed_dominated = true
     ∧ check_to_unbox no_local kept_result_unboxed_facts
         kept_result_unboxed_tables kept_result_cannot_change size false
         to_unbox = true
     ∧ check_to_change_representation no_local kept_result_unboxed_facts
         kept_result_unboxed_tables kept_result_cannot_change size false
         to_unbox tcr = true
     ∧ check_sets_whole kept_result_unboxed_facts kept_result_unboxed_tables
         to_unbox tcr = true)
  ∧ behaves (FlambdaReaperTupledSpec.on_its_own kept_result_unboxed)
      kept_result_exit
  ∧ behaves (FlambdaReaperTupledSpec.on_its_own (kept_result_rebuilt false))
      Goes_wrong
  ∧ behaves (FlambdaReaperTupledSpec.on_its_own (kept_result_rebuilt true))
      kept_result_exit
  ∧ check_result_receivers kept_result_unboxed_facts
      kept_result_unboxed_tables kept_result_unboxed_dominated
      (kept_result_to_unbox false) = false
  ∧ check_result_receivers kept_result_unboxed_facts
      kept_result_unboxed_tables kept_result_unboxed_dominated
      (kept_result_to_unbox true) = true
  ∧ check_result_receivers unboxed_closure_facts unboxed_closure_tables
      unboxed_closure_dominated [v 60] = true
  ∧ check_result_receivers read_block_facts read_block_tables
      [(v 10, v 10)] [v 10] = true
  ∧ check_result_receivers block_to_continuation_facts
      block_to_continuation_tables [(v 10, v 10); (v 10, v 30)] [v 10]
    = true
  ∧ check_result_receivers block_from_code_facts block_from_code_tables
      block_from_code_dominated [v 42] = true
  ∧ check_result_receivers nested_block_facts nested_block_tables
      [(v 11, v 11); (v 10, v 10); (v 10, v 12)] [v 11; v 10] = true.

(* Why. Without the fix, the block [x42] is unboxed, and dominates the
   result 3 of the code dep of 50, which the code then unboxes; [x32],
   which the call that returns to 21 reads that result into, has two
   allocation points and is not unboxed: 21 keeps it. With the fix,
   [x42] is left out of [to_unbox], and the rest is as it was. *)
Definition kept_result_unboxed_cause_statement : Prop :=
  kept_result_to_unbox false = [v 42]
  ∧ mem_pair (v 42) (m 3) kept_result_unboxed_dominated = true
  ∧ multiple_allocation_points kept_result_unboxed_tables (v 32) = true
  ∧ kept_result_to_unbox true = []
  ∧ kept_result_tcr true = kept_result_tcr false.

(* ======================================================================== *)
(* A fixed miscompile: a result unboxed by code and deleted by a            *)
(* continuation                                                             *)
(* ======================================================================== *)

(* The unit of testsuite/tests/reaper/unbox_dead.ml, with a block for the
   boxed float: the code 50 returns a block of its parameter and 8. The
   unit calls it returning to 20, which reads field 0 of the block into
   [x31]. The arm 1 of a [Switch], taken at run time, calls it again
   returning to 21, as the arm 0 jumps to 21 with 3; 21 does not use its
   parameter [x32], and returns [x31]. *)
Definition unbox_dead_unit : expr :=
  Let_static [Bind_code 50 code_50]
    (Let 60 Value (Closure_of_set 70 [(70, 50)] [])
       (Let_cont_non_recursive 20
          (Handler [30] (make_handler_metadata [Value])
             (Let 31 Value (load 0 (Var 30))
                (Let_cont_non_recursive 21
                   (Handler [32] (make_handler_metadata [Value])
                      (return_to 1 [Var 31]))
                   (Let_cont_non_recursive 26
                      (Handler [] no_params (return_to 21 [int 3]))
                      (Let_cont_non_recursive 27
                         (Handler [] no_params (Apply (call_50 [int 9] 21)))
                         (Switch (Const (Naked_immediate 1))
                            [(0%Z, goto 26); (1%Z, goto 27)]))))))
          (Apply (call_50 [int 9] 20)))).

Definition unbox_dead_facts : facts :=
  traverse_unit Auto interface unbox_dead_unit.

(* The solution of the unit, as the solver of the test tool computes it.
   The code dep of 50 is as in [kept_result_unboxed]. [x32] has the
   sources 3, a constant, and the result 3 of the code dep, which the
   block [x42] dominates; [x32] has no use. *)
Definition unbox_dead_tables : tables :=
  {| t_alias := [(m 18, m 7); (m 5, m 17); (m 13, m 7); (m 5, m 12);
       (m 14, m 16); (m 9, m 11); (v 32, consts); (m 1, v 31); (m 3, v 42);
       (v 41, m 5); (v 40, m 6); (m 11, m 7); (m 16, m 7); (v 30, m 3);
       (v 32, m 3); (m 2, m 4); (m 15, c 50); (m 10, c 50); (m 6, consts);
       (v 31, v 40); (m 5, v 60)];
     t_any_usage := [m 1; m 15; consts; m 10; m 2; v 31; m 4; c 50; v 40; m 6];
     t_any_source := [c 50; consts; v 32; m 15; m 10; m 6; v 40; v 31; m 1];
     t_has_usage := [m 1; m 15; consts; m 10; m 2; v 31; m 14; m 9; v 30; m 16;
       m 11; m 7; m 3; m 4; c 50; v 40; v 42; m 6; v 60];
     t_has_source := [c 50; consts; v 32; v 60; v 42; m 8; m 7; m 18; m 13;
       m 3; m 11; m 16; v 30; m 15; m 10; m 6; m 14; m 9; v 40; v 31; m 5; m 1;
       v 41];
     t_usages := [(v 60, v 60); (m 14, m 14); (m 9, m 9); (v 30, v 30);
       (m 16, m 14); (m 11, m 9); (m 7, m 9); (m 7, m 14); (m 3, v 30);
       (v 42, v 30)];
     t_sources := [(v 60, v 60); (v 42, v 42); (m 8, m 8); (m 7, m 7);
       (m 18, m 7); (m 13, m 7); (m 3, v 42); (m 11, m 7); (m 16, m 7);
       (m 14, m 7); (m 9, m 7); (v 30, v 42); (m 5, v 60); (v 41, v 60)];
     t_reading_field := [];
     t_escaping_field := [];
     t_field_used := [(v 60, known); (v 42, Field_block 0 Value);
       (m 7, result 0); (m 7, Field_return_of_call Exn);
       (m 7, Field_code_id_of_call_witness)] |}.

Definition unbox_dead_cannot_change : list node :=
  derive_cannot_change unbox_dead_facts unbox_dead_tables.

Definition unbox_dead_dominated : list (node * node) :=
  derive_dominated unbox_dead_tables.

Definition unbox_dead_to_unbox : list node :=
  derive_to_unbox no_local unbox_dead_facts unbox_dead_tables
    unbox_dead_cannot_change max_unbox_size false.

Definition unbox_dead_tcr : list node :=
  derive_to_change_representation no_local unbox_dead_facts
    unbox_dead_tables unbox_dead_cannot_change max_unbox_size false
    unbox_dead_to_unbox.

Definition unbox_dead_code_deps : finite_map code_dep :=
  unit_code_deps Auto interface unbox_dead_unit.

Definition unbox_dead_rebuilt : expr :=
  rebuild_unit_unbox_repr no_local unbox_dead_facts unbox_dead_tables
    unbox_dead_cannot_change unbox_dead_code_deps unbox_dead_dominated
    unbox_dead_to_unbox unbox_dead_tcr true unbox_dead_unit.

Definition unbox_dead_exit : behaviour :=
  Exits 1 [Value_const (Tagged_immediate 9)] empty_heap.

(* [rev_args_or_invalid] of [make_apply_wrapper] before the fix, the
   match of [wrapper_args_u] in its old order: a result that the callee
   unboxes and the continuation deletes is [None] ([Invalid]). *)
Fixpoint wrapper_args_u_before_fix (apply_decisions : list decision)
    (func : list (wrapper_decision * flambda_kind)) : option (list simple) :=
  match func with
  | [] => Some []
  | (fd, kind) :: func =>
    let '(ad, apply_decisions) :=
      match apply_decisions with
      | [] => (Decided (Keep kind), [])
      | ad :: apply_decisions => (ad, apply_decisions)
      end
    in
    match wrapper_args_u_before_fix apply_decisions func with
    | None => None
    | Some rest =>
      match ad, fd with
      | Unbox _, (Wrapper_keep _ | Wrapper_delete)
      | Decided _, Wrapper_unbox _ => None
      | Decided Delete, _ => Some rest
      | Decided (Keep _), Wrapper_keep x => Some (Var x :: rest)
      | Decided (Keep kind), Wrapper_delete =>
        Some (Const (Poison kind) :: rest)
      | Unbox fields_apply, Wrapper_unbox fields_func =>
        match
          unboxed_field_pairs Value (Unboxed fields_apply)
            (Unboxed fields_func)
        with
        | Some pairs => Some (map (fun '(_, x, _) => Var x) pairs ++ rest)
        | None => None
        end
      end
    end
  end.

(* The decisions of the callee 50 for its result, freshened for the
   wrapper of the call that returns to 21, whose decision is [Delete]. *)
Definition unbox_dead_wrapper_decisions
    : list (wrapper_decision * flambda_kind) :=
  freshen_decisions [(Unbox [(Field_block 0 Value, Not_unboxed 62)], Value)]
    64.

(* The calls of the rebuilt unit, and the rebuilt unit with [h] for the
   handler of the wrapper 64 of the call that returns to 21: that handler
   jumps to 21 with no argument since the fix, and was [Invalid] before
   it. The closure [x60] changes its representation, to the function
   slot 71; the code returns the variable 61 of the field of its result,
   and 20 has the variable 63 of the field of [x30] for parameter. *)
Definition unbox_dead_call (k : continuation) : expr :=
  Apply
    {| ap_callee := Var 60; ap_call_kind := Direct 50; ap_args := [int 9];
       ap_continuation := k; ap_exn_continuation := 2;
       ap_exn_extra_args := [];
       ap_args_arity := [Unboxed_product [Singleton Value]];
       ap_return_arity := [Unboxed_product [Singleton Value]] |}.

Definition unbox_dead_rebuilt_with (h : expr) : expr :=
  Let_static [Bind_code 50 code_50_unboxed]
    (Let 60 Value (Closure_of_set 71 [(71, 50)] [])
       (Let_cont_non_recursive 20
          (Handler [63] (make_handler_metadata [Value])
             (Let 31 Value (Simple (Var 63))
                (Let_cont_non_recursive 21
                   (Handler [] no_params (return_to 1 [Var 31]))
                   (Let_cont_non_recursive 26
                      (Handler [] no_params (return_to 21 []))
                      (Let_cont_non_recursive 27
                         (Handler [] no_params
                            (Let_cont_non_recursive 64
                               (Handler [64] (make_handler_metadata [Value])
                                  h)
                               (unbox_dead_call 64)))
                         (Switch (Const (Naked_immediate 1))
                            [(0%Z, goto 26); (1%Z, goto 27)]))))))
          (unbox_dead_call 20))).

(* The tables pass every check; [x42] is unboxed, and so is the result of
   the code, which it dominates; [x32], of two sources, is not, and has
   no use. The unit exits with 9, and so does it rebuilt. The rebuilt
   unit is [unbox_dead_rebuilt_with] the jump to 21: before the fix,
   [rev_args_or_invalid] had no arguments for the call that returns to
   21, whose wrapper was then [Invalid], and the rebuilt unit went
   wrong. *)
Definition unbox_dead_statement : Prop :=
  check_closed no_local unbox_dead_facts unbox_dead_tables = true
  ∧ check_field_used_exact no_local unbox_dead_facts unbox_dead_tables
    = true
  ∧ check_cc_closed unbox_dead_facts unbox_dead_tables
      unbox_dead_cannot_change = true
  ∧ check_dominated unbox_dead_tables unbox_dead_dominated = true
  ∧ check_to_unbox no_local unbox_dead_facts unbox_dead_tables
      unbox_dead_cannot_change max_unbox_size false unbox_dead_to_unbox
    = true
  ∧ check_to_change_representation no_local unbox_dead_facts
      unbox_dead_tables unbox_dead_cannot_change max_unbox_size false
      unbox_dead_to_unbox unbox_dead_tcr = true
  ∧ check_sets_whole unbox_dead_facts unbox_dead_tables unbox_dead_to_unbox
      unbox_dead_tcr = true
  ∧ unbox_dead_to_unbox = [v 42]
  ∧ mem_pair (v 42) (m 3) unbox_dead_dominated = true
  ∧ multiple_allocation_points unbox_dead_tables (v 32) = true
  ∧ mem_node (v 32) (t_has_usage unbox_dead_tables) = false
  ∧ unit_calling_convention_changes_u unbox_dead_facts unbox_dead_tables
      unbox_dead_cannot_change unbox_dead_code_deps unbox_dead_dominated
      unbox_dead_to_unbox unbox_dead_unit
    = [(50, Changing_u [Decided (Keep Value)]
              [Unbox [(Field_block 0 Value, Not_unboxed 62)]])]
  ∧ behaves (FlambdaReaperTupledSpec.on_its_own unbox_dead_unit)
      unbox_dead_exit
  ∧ unbox_dead_rebuilt = unbox_dead_rebuilt_with (return_to 21 [])
  ∧ behaves (FlambdaReaperTupledSpec.on_its_own unbox_dead_rebuilt)
      unbox_dead_exit
  ∧ wrapper_args_u [Decided Delete] unbox_dead_wrapper_decisions = Some []
  ∧ wrapper_args_u_before_fix [Decided Delete] unbox_dead_wrapper_decisions
    = None
  ∧ behaves
      (FlambdaReaperTupledSpec.on_its_own (unbox_dead_rebuilt_with Invalid))
      Goes_wrong.

(* ======================================================================== *)
(* The rules with seeds of a fix                                            *)
(* ======================================================================== *)

(* The tables of the rules with the allocations [seeds] added to
   [cannot_unbox0], as a rule that concludes it from relations other than
   itself adds them ([derived_cannot_unbox0_seeds]): before the rounds of
   [cu0_stored], which are made again. The other tables are those of
   [unboxing_relations], which no seed of [cannot_unbox0] changes. *)
Definition unboxing_relations_with (is_local : field -> bool) (F : facts)
    (T : tables) (cannot_change : list node) (max_unbox_size : nat)
    (lambda_lifting : bool) (seeds : list node) : unboxing_tables :=
  let R :=
    unboxing_relations is_local F T cannot_change max_unbox_size
      lambda_lifting
  in
  let cu0 :=
    stored_rounds F T (ut_crr R) (S (length (t_sources T)))
      (Keyed.dedup node_eqb (seeds ++ ut_cu0 R))
  in
  {| ut_crr0 := ut_crr0 R; ut_crr1 := ut_crr1 R; ut_crr := ut_crr R;
     ut_cu0 := cu0;
     ut_cu := Keyed.dedup node_eqb (cu0 ++ by_function_slots F cu0) |}.

(* [derive_to_unbox] for those tables. *)
Definition derive_to_unbox_with (is_local : field -> bool) (F : facts)
    (T : tables) (cannot_change : list node) (max_unbox_size : nat)
    (lambda_lifting : bool) (seeds : list node) : list node :=
  let R :=
    unboxing_relations_with is_local F T cannot_change max_unbox_size
      lambda_lifting seeds
  in
  filter (fun x => negb (mem_node x (ut_cu R)))
    (Keyed.dedup node_eqb (t_has_usage T)).

(* ======================================================================== *)
(* A counterexample: a field read immutably and mutably                     *)
(* ======================================================================== *)

(* The fix, as what it adds to [cannot_unbox0]: an allocation with a
   source that is a source of a field of [base] ([constructor]), which a
   use of [base] reads into a name [v] with a use that the allocation
   does not dominate. *)
Definition field_reader_seeds (F : facts) (T : tables)
    (dominated : list (node * node)) : list node :=
  flat_map
    (fun '(base, relation, alias) =>
       flat_map
         (fun '(alias', allocation_id) =>
            if node_eqb alias alias'
               && mem_node allocation_id (t_has_source T)
               && existsb
                    (fun '(v, relation', usage) =>
                       field_eqb relation relation'
                       && mem_pair base usage (t_usages T)
                       && mem_node v (t_has_usage T)
                       && negb (mem_pair allocation_id v dominated))
                    (f_accessor F)
            then [allocation_id]
            else [])
         (t_sources T))
    (f_constructor F).

(* The fix, as what it asks of a table of [to_unbox]: no allocation that
   it adds to [cannot_unbox0] is unboxed. *)
Definition check_field_readers (F : facts) (T : tables)
    (dominated : list (node * node)) (to_unbox : list node) : bool :=
  forallb (fun x => negb (mem_node x to_unbox))
    (field_reader_seeds F T dominated).

(* [Block_load] of a value from a mutable field. *)
Definition load_mutable (field : Z) (block : simple) : named :=
  Prim (Unary (Block_load (Block_access_values None None Any_value)
                 Mutable field)) [block].

(* A call of [callee] with 3, from code that returns to 3 and raises to
   4. *)
Definition call_with_3 (callee : variable) : apply_expr :=
  {| ap_callee := Var callee; ap_call_kind := Indirect; ap_args := [int 3];
     ap_continuation := 3; ap_exn_continuation := 4;
     ap_exn_extra_args := [];
     ap_args_arity := [Singleton Value];
     ap_return_arity := [Singleton Value] |}.

(* The unit of mixed_mutable_field.fl, without [g] and the module block:
   the code 50 is [f] without its parameter [w], whose [x] is [x40], and
   the [Switch] on a constant stands for the match on [w], whose arm 26,
   [A], is taken. The arm 26 reads field 0 of [x40] immutably into [x42],
   [*match*], and field 0 of that into [x43], [p], which it returns; the
   arm 27, [B], reads field 0 of [x40] mutably into [x44], [x.b], and
   calls it with 3. The unit calls 50 through [x60] with [x11], a block
   of the pair [x10] of 3 and 3, as [g] does, and returns what it
   returns. *)
Definition mixed_field_code : code :=
  Code [40] 41 3 4 (make_code_metadata [Singleton Value] [Singleton Value])
    (Let_cont_non_recursive 26
       (Handler [] no_params
          (Let 42 Value (load 0 (Var 40))
             (Let 43 Value (load 0 (Var 42)) (return_to 3 [Var 43]))))
       (Let_cont_non_recursive 27
          (Handler [] no_params
             (Let 44 Value (load_mutable 0 (Var 40))
                (Apply (call_with_3 44))))
          (Switch (Const (Naked_immediate 0))
             [(0%Z, goto 26); (1%Z, goto 27)]))).

Definition mixed_field_unit : expr :=
  Let_static [Bind_code 50 mixed_field_code]
    (Let 60 Value (Closure_of_set 70 [(70, 50)] [])
       (Let 10 Value (make_block [int 3; int 3])
          (Let 11 Value (make_block [Var 10])
             (Apply (call_50 [Var 11] 1))))).

Definition mixed_field_facts : facts :=
  traverse_unit Auto interface mixed_field_unit.

(* The solution of the unit, as the solver of the test tool computes it.
   The code dep of 50 has the result 3, [my_closure] 5, the parameter 6
   and the witness 7 of known arity. [x40] has the source [x11], [x42]
   the source [x10]; [x44], the result of a mutable load, has any source,
   and is a use of [x10] too. *)
Definition mixed_field_tables : tables :=
  {| t_alias := [(m 16, m 7); (m 5, m 15); (m 12, m 14); (m 3, v 43);
       (v 41, m 5); (v 40, m 6); (m 14, m 7); (m 1, m 3); (m 2, m 4);
       (m 13, c 50); (m 6, v 11); (v 42, v 10); (v 44, v 10); (v 43, consts);
       (m 5, v 60); (m 10, m 11)];
     t_any_usage := [m 1; m 13; m 9; m 2; consts; m 3; m 4; c 50; v 43];
     t_any_source := [v 44; c 50; consts; m 11; m 13; v 43; m 3; m 1; m 10;
       m 4; m 9; m 2];
     t_has_usage := [m 1; m 13; m 9; m 2; m 12; m 10; m 14; m 7; m 3; m 4;
       c 50; consts; v 43; m 11; v 60; v 44; v 40; v 42; m 6; v 11; v 10];
     t_has_source := [v 44; c 50; consts; m 11; v 11; v 10; v 60; m 8; m 7;
       m 16; m 14; m 13; m 6; v 42; v 43; m 12; m 3; v 40; m 1; m 5; m 10; m 4;
       m 9; v 41; m 2];
     t_usages := [(v 60, v 60); (m 12, m 12); (v 44, v 44); (m 10, m 10);
       (v 40, v 40); (v 42, v 42); (m 14, m 12); (m 6, v 40); (m 7, m 12);
       (v 11, v 40); (v 10, v 42); (v 10, v 44); (m 11, m 10)];
     t_sources := [(v 60, v 60); (v 11, v 11); (v 10, v 10); (m 8, m 8);
       (m 7, m 7); (m 16, m 7); (m 14, m 7); (m 12, m 7); (m 6, v 11);
       (v 40, v 11); (v 42, v 10); (m 5, v 60); (v 41, v 60)];
     t_reading_field := [];
     t_escaping_field := [];
     t_field_used := [(v 60, known); (v 11, Field_block 0 Value);
       (v 10, Field_block 0 Value); (m 7, result 0);
       (m 7, Field_return_of_call Exn);
       (m 7, Field_code_id_of_call_witness)] |}.

Definition mixed_field_cannot_change : list node :=
  derive_cannot_change mixed_field_facts mixed_field_tables.

Definition mixed_field_dominated : list (node * node) :=
  derive_dominated mixed_field_tables.

Definition mixed_field_code_deps : finite_map code_dep :=
  unit_code_deps Auto interface mixed_field_unit.

(* The seeds of the fix in the unit. *)
Definition mixed_field_seeds : list node :=
  field_reader_seeds mixed_field_facts mixed_field_tables
    mixed_field_dominated.

(* The tables of [to_unbox] and [to_change_representation] that the rules
   give, without the fix and with its seeds, and the unit rebuilt with
   them. *)
Definition mixed_field_to_unbox (fixed : bool) : list node :=
  if fixed
  then
    derive_to_unbox_with no_local mixed_field_facts mixed_field_tables
      mixed_field_cannot_change max_unbox_size false mixed_field_seeds
  else
    derive_to_unbox no_local mixed_field_facts mixed_field_tables
      mixed_field_cannot_change max_unbox_size false.

Definition mixed_field_tcr (fixed : bool) : list node :=
  derive_to_change_representation no_local mixed_field_facts
    mixed_field_tables mixed_field_cannot_change max_unbox_size false
    (mixed_field_to_unbox fixed).

Definition mixed_field_rebuilt (fixed : bool) : expr :=
  rebuild_unit_unbox_repr no_local mixed_field_facts mixed_field_tables
    mixed_field_cannot_change mixed_field_code_deps mixed_field_dominated
    (mixed_field_to_unbox fixed) (mixed_field_tcr fixed) true
    mixed_field_unit.

Definition mixed_field_exit : behaviour :=
  Exits 1 [Value_const (Tagged_immediate 3)] empty_heap.

(* The tables pass every check. The unit exits with 3; rebuilt, it goes
   wrong. The check of the fix rejects the tables. With the seeds of the
   fix, the tables are closed under the rules, pass the check of the fix
   and that of sets, and the unit rebuilt with them exits with 3. The
   check passes the tables of [unboxed_closure], of the four examples of
   blocks of FlambdaReaperUnboxing.v, of [unbox_dead_unit], which
   [check_result_receivers] rejects, and of [dead_callee_unit] with lambda
   lifting. *)
Definition mixed_mutable_field_statement : Prop :=
  check_closed no_local mixed_field_facts mixed_field_tables = true
  ∧ check_field_used_exact no_local mixed_field_facts mixed_field_tables
    = true
  ∧ check_cc_closed mixed_field_facts mixed_field_tables
      mixed_field_cannot_change = true
  ∧ check_dominated mixed_field_tables mixed_field_dominated = true
  ∧ check_to_unbox no_local mixed_field_facts mixed_field_tables
      mixed_field_cannot_change max_unbox_size false
      (mixed_field_to_unbox false) = true
  ∧ check_to_change_representation no_local mixed_field_facts
      mixed_field_tables mixed_field_cannot_change max_unbox_size false
      (mixed_field_to_unbox false) (mixed_field_tcr false) = true
  ∧ check_sets_whole mixed_field_facts mixed_field_tables
      (mixed_field_to_unbox false) (mixed_field_tcr false) = true
  ∧ behaves (FlambdaReaperTupledSpec.on_its_own mixed_field_unit)
      mixed_field_exit
  ∧ behaves (FlambdaReaperTupledSpec.on_its_own (mixed_field_rebuilt false))
      Goes_wrong
  ∧ check_field_readers mixed_field_facts mixed_field_tables
      mixed_field_dominated (mixed_field_to_unbox false) = false
  ∧ check_unboxing_closed no_local mixed_field_facts mixed_field_tables
      mixed_field_cannot_change max_unbox_size false
      (unboxing_relations_with no_local mixed_field_facts mixed_field_tables
         mixed_field_cannot_change max_unbox_size false mixed_field_seeds)
    = true
  ∧ check_field_readers mixed_field_facts mixed_field_tables
      mixed_field_dominated (mixed_field_to_unbox true) = true
  ∧ check_sets_whole mixed_field_facts mixed_field_tables
      (mixed_field_to_unbox true) (mixed_field_tcr true) = true
  ∧ behaves (FlambdaReaperTupledSpec.on_its_own (mixed_field_rebuilt true))
      mixed_field_exit
  ∧ check_field_readers unboxed_closure_facts unboxed_closure_tables
      unboxed_closure_dominated [v 60] = true
  ∧ check_field_readers read_block_facts read_block_tables [(v 10, v 10)]
      [v 10] = true
  ∧ check_field_readers block_to_continuation_facts
      block_to_continuation_tables [(v 10, v 10); (v 10, v 30)] [v 10]
    = true
  ∧ check_field_readers block_from_code_facts block_from_code_tables
      block_from_code_dominated [v 42] = true
  ∧ check_field_readers nested_block_facts nested_block_tables
      [(v 11, v 11); (v 10, v 10); (v 10, v 12)] [v 11; v 10] = true
  ∧ check_field_readers unbox_dead_facts unbox_dead_tables
      unbox_dead_dominated unbox_dead_to_unbox = true
  ∧ check_result_receivers unbox_dead_facts unbox_dead_tables
      unbox_dead_dominated unbox_dead_to_unbox = false
  ∧ check_field_readers dead_callee_facts dead_callee_tables
      dead_callee_dominated (dead_callee_to_unbox true) = true.

(* Why. Without the fix, the block [x11] and the pair [x10] are unboxed:
   [x10] dominates [x42], and [x11] dominates [x40], which the code
   unboxes to its field 0, a value; [x44], of any source, which reads the
   same field 0, is dominated by nothing and kept. The seeds of the fix
   are [x10]; with them, [x11] alone is unboxed, and the rest is as it
   was. *)
Definition mixed_mutable_field_cause_statement : Prop :=
  mixed_field_to_unbox false = [v 11; v 10]
  ∧ mem_pair (v 10) (v 42) mixed_field_dominated = true
  ∧ mem_pair (v 11) (v 40) mixed_field_dominated = true
  ∧ unit_calling_convention_changes_u mixed_field_facts mixed_field_tables
      mixed_field_cannot_change mixed_field_code_deps mixed_field_dominated
      (mixed_field_to_unbox false) mixed_field_unit
    = [(50, Changing_u [Unbox [(Field_block 0 Value, Not_unboxed 63)]]
              [Decided (Keep Value)])]
  ∧ mem_node (v 44) (t_any_source mixed_field_tables) = true
  ∧ multiple_allocation_points mixed_field_tables (v 44) = true
  ∧ mixed_field_seeds = [v 10]
  ∧ mixed_field_to_unbox true = [v 11]
  ∧ mixed_field_tcr true = mixed_field_tcr false.

(* ======================================================================== *)
(* A counterexample: a block given to code that keeps its calling           *)
(* convention                                                               *)
(* ======================================================================== *)

(* The fix, as what it adds to [cannot_unbox0]: an allocation with a
   source that is a source of an argument of a call ([argument]) that may
   call code that cannot change its calling convention, a source of the
   callee of the call. *)
Definition escaping_argument_seeds (F : facts) (T : tables)
    (cannot_change : list node) : list node :=
  flat_map
    (fun '(from, _, base) =>
       if existsb
            (fun '(base', call_witness) =>
               node_eqb base base'
               && witness_cannot_change F cannot_change call_witness)
            (t_sources T)
       then
         flat_map
           (fun '(from', allocation_id) =>
              if node_eqb from from'
                 && mem_node allocation_id (t_has_source T)
              then [allocation_id]
              else [])
           (t_sources T)
       else [])
    (f_argument F).

(* The fix, as what it asks of a table of [to_unbox]: no allocation that
   it adds to [cannot_unbox0] is unboxed. *)
Definition check_escaping_arguments (F : facts) (T : tables)
    (cannot_change : list node) (to_unbox : list node) : bool :=
  forallb (fun x => negb (mem_node x to_unbox))
    (escaping_argument_seeds F T cannot_change).

(* The unit of escaping_parameter_argument.fl, without [g]: the code 50
   is [f] without its parameter [w], whose [p] is [x40], and the [Switch]
   on a constant stands for the match on [w], whose arm 26, [P], is
   taken. The arm 26 returns 0; the arm 27, [F], calls [x40] with 3. The
   unit calls 50 through [x60] with the pair [x10] of 5 and 5, returning
   to 20, which returns a block of [x60] and the result, as the module
   block holds [f] and [r]: [x60] escapes, and the code 50 cannot change
   its calling convention. [escaping_argument_with e] has [e] in the
   scope of 20. *)
Definition escaping_argument_code : code :=
  Code [40] 41 3 4 (make_code_metadata [Singleton Value] [Singleton Value])
    (Let_cont_non_recursive 26
       (Handler [] no_params (return_to 3 [int 0]))
       (Let_cont_non_recursive 27
          (Handler [] no_params (Apply (call_with_3 40)))
          (Switch (Const (Naked_immediate 0))
             [(0%Z, goto 26); (1%Z, goto 27)]))).

Definition escaping_argument_with (e : expr) : expr :=
  Let_static [Bind_code 50 escaping_argument_code]
    (Let 60 Value (Closure_of_set 70 [(70, 50)] [])
       (Let_cont_non_recursive 20
          (Handler [30] (make_handler_metadata [Value])
             (Let 31 Value (make_block [Var 60; Var 30])
                (return_to 1 [Var 31])))
          e)).

Definition escaping_argument_unit : expr :=
  escaping_argument_with
    (Let 10 Value (make_block [int 5; int 5])
       (Apply (call_50 [Var 10] 20))).

Definition escaping_argument_facts : facts :=
  traverse_unit Auto interface escaping_argument_unit.

(* The solution of the unit, as the solver of the test tool computes it.
   The code dep of 50 has the parameter 6 and the witness 7 of known
   arity, of any usage; the call is 12, whose argument is [x10]. The
   parameter 6 and [x40] have any source, and no source recorded; 6 is
   an alias of [x10]. *)
Definition escaping_argument_tables : tables :=
  {| t_alias := [(m 16, m 7); (m 5, m 15); (m 12, m 14); (m 1, v 31);
       (m 3, consts); (v 41, m 5); (v 40, m 6); (m 5, v 60); (m 10, m 11);
       (m 14, m 7); (v 30, m 3); (m 2, m 4); (m 13, c 50); (m 6, v 10)];
     t_any_usage := [m 1; m 13; m 9; m 2; v 31; v 60; v 30; consts; m 7; m 8;
       m 4; m 3; c 50];
     t_any_source := [c 50; consts; m 3; m 6; v 40; m 11; m 10; m 4; m 9; v 30;
       m 2; m 13];
     t_has_usage := [m 1; m 13; m 9; m 2; v 31; v 60; v 30; consts; m 7; m 8;
       m 4; m 3; c 50; m 12; m 10; m 14; m 11; v 40; m 6; v 10];
     t_has_source := [c 50; consts; m 3; m 6; v 40; m 11; m 10; m 4; m 9; v 10;
       v 31; v 60; m 8; m 7; m 16; m 1; m 5; m 14; v 30; m 2; m 13; m 12;
       v 41];
     t_usages := [(m 12, m 12); (v 40, v 40); (m 10, m 10); (m 14, m 12);
       (m 6, v 40); (m 11, m 10); (v 10, v 40)];
     t_sources := [(v 60, v 60); (v 10, v 10); (v 31, v 31); (m 8, m 8);
       (m 7, m 7); (m 16, m 7); (m 1, v 31); (m 5, v 60); (v 41, v 60);
       (m 14, m 7); (m 12, m 7)];
     t_reading_field := [];
     t_escaping_field := [];
     t_field_used := [(v 60, known); (v 60, unknown);
       (v 31, Field_block 0 Value); (v 31, Field_block 1 Value);
       (v 31, Field_is_int); (v 31, Field_get_tag);
       (v 60, Field_function_slot 70); (m 8, Field_return_of_call Exn);
       (m 8, result 0); (m 8, Field_code_id_of_call_witness); (m 7, result 0);
       (m 7, Field_return_of_call Exn);
       (m 7, Field_code_id_of_call_witness)] |}.

Definition escaping_argument_cannot_change : list node :=
  derive_cannot_change escaping_argument_facts escaping_argument_tables.

Definition escaping_argument_dominated : list (node * node) :=
  derive_dominated escaping_argument_tables.

Definition escaping_argument_code_deps : finite_map code_dep :=
  unit_code_deps Auto interface escaping_argument_unit.

(* The seeds of the fix in the unit. *)
Definition escaping_argument_seeds_of_unit : list node :=
  escaping_argument_seeds escaping_argument_facts escaping_argument_tables
    escaping_argument_cannot_change.

(* The tables of [to_unbox] and [to_change_representation] that the rules
   give, without the fix and with its seeds, and the unit rebuilt with
   them. *)
Definition escaping_argument_to_unbox (fixed : bool) : list node :=
  if fixed
  then
    derive_to_unbox_with no_local escaping_argument_facts
      escaping_argument_tables escaping_argument_cannot_change
      max_unbox_size false escaping_argument_seeds_of_unit
  else
    derive_to_unbox no_local escaping_argument_facts escaping_argument_tables
      escaping_argument_cannot_change max_unbox_size false.

Definition escaping_argument_tcr (fixed : bool) : list node :=
  derive_to_change_representation no_local escaping_argument_facts
    escaping_argument_tables escaping_argument_cannot_change max_unbox_size
    false (escaping_argument_to_unbox fixed).

Definition escaping_argument_rebuilt (fixed : bool) : expr :=
  rebuild_unit_unbox_repr no_local escaping_argument_facts
    escaping_argument_tables escaping_argument_cannot_change
    escaping_argument_code_deps escaping_argument_dominated
    (escaping_argument_to_unbox fixed) (escaping_argument_tcr fixed) true
    escaping_argument_unit.

Definition escaping_argument_exit : behaviour :=
  Exits 1
    [Value_block 0
       [Value_closure_of_set 70 [(70, 50)] [];
        Value_const (Tagged_immediate 0)]]
    empty_heap.

(* The tables pass every check. The unit exits with a block of the
   closure and 0; rebuilt, it goes wrong. The check of the fix rejects
   the tables. With the seeds of the fix, the tables are closed under the
   rules, pass the check of the fix and that of sets, and the unit
   rebuilt with them exits as the unit does. The rebuilt unit is the unit
   without the [Let] of the pair, whose variable it gives the call; with
   poison for argument, the other half of the fix, it exits as the unit
   does. The check passes the tables of [unboxed_closure], of
   [block_from_code], of [kept_result_unboxed] with its fix, of
   [unbox_dead_unit] and of [dead_callee_unit] with lambda lifting, the
   examples whose code cannot change its calling convention or whose
   calls give an allocation. *)
Definition escaping_parameter_argument_statement : Prop :=
  check_closed no_local escaping_argument_facts escaping_argument_tables
    = true
  ∧ check_field_used_exact no_local escaping_argument_facts
      escaping_argument_tables = true
  ∧ check_cc_closed escaping_argument_facts escaping_argument_tables
      escaping_argument_cannot_change = true
  ∧ check_dominated escaping_argument_tables escaping_argument_dominated
    = true
  ∧ check_to_unbox no_local escaping_argument_facts escaping_argument_tables
      escaping_argument_cannot_change max_unbox_size false
      (escaping_argument_to_unbox false) = true
  ∧ check_to_change_representation no_local escaping_argument_facts
      escaping_argument_tables escaping_argument_cannot_change
      max_unbox_size false (escaping_argument_to_unbox false)
      (escaping_argument_tcr false) = true
  ∧ check_sets_whole escaping_argument_facts escaping_argument_tables
      (escaping_argument_to_unbox false) (escaping_argument_tcr false)
    = true
  ∧ behaves (FlambdaReaperTupledSpec.on_its_own escaping_argument_unit)
      escaping_argument_exit
  ∧ behaves
      (FlambdaReaperTupledSpec.on_its_own (escaping_argument_rebuilt false))
      Goes_wrong
  ∧ check_escaping_arguments escaping_argument_facts escaping_argument_tables
      escaping_argument_cannot_change (escaping_argument_to_unbox false)
    = false
  ∧ check_unboxing_closed no_local escaping_argument_facts
      escaping_argument_tables escaping_argument_cannot_change
      max_unbox_size false
      (unboxing_relations_with no_local escaping_argument_facts
         escaping_argument_tables escaping_argument_cannot_change
         max_unbox_size false escaping_argument_seeds_of_unit)
    = true
  ∧ check_escaping_arguments escaping_argument_facts escaping_argument_tables
      escaping_argument_cannot_change (escaping_argument_to_unbox true)
    = true
  ∧ check_sets_whole escaping_argument_facts escaping_argument_tables
      (escaping_argument_to_unbox true) (escaping_argument_tcr true) = true
  ∧ behaves
      (FlambdaReaperTupledSpec.on_its_own (escaping_argument_rebuilt true))
      escaping_argument_exit
  ∧ escaping_argument_rebuilt false
    = escaping_argument_with (Apply (call_50 [Var 10] 20))
  ∧ behaves
      (FlambdaReaperTupledSpec.on_its_own
         (escaping_argument_with (Apply (call_50 [Const (Poison Value)] 20))))
      escaping_argument_exit
  ∧ check_escaping_arguments unboxed_closure_facts unboxed_closure_tables []
      [v 60] = true
  ∧ check_escaping_arguments block_from_code_facts block_from_code_tables
      (derive_cannot_change block_from_code_facts block_from_code_tables)
      [v 42] = true
  ∧ check_escaping_arguments kept_result_unboxed_facts
      kept_result_unboxed_tables kept_result_cannot_change
      (kept_result_to_unbox true) = true
  ∧ check_escaping_arguments unbox_dead_facts unbox_dead_tables
      unbox_dead_cannot_change unbox_dead_to_unbox = true
  ∧ check_escaping_arguments dead_callee_facts dead_callee_tables
      dead_callee_cannot_change (dead_callee_to_unbox true) = true.

(* Why. Without the fix, the pair [x10] is unboxed, to no field. The code
   50 cannot change its calling convention, and its parameter 6, of
   which [x10] is an alias, has any source and no source recorded, so
   that [cu0_parameter] does not hold of [x10]. The call does not change
   its calling convention, and the rebuilt unit has [x10] free. The seeds
   of the fix are [x10]; with them, nothing is unboxed. Neither check
   rejects the tables of the other counterexample. *)
Definition escaping_parameter_argument_cause_statement : Prop :=
  escaping_argument_to_unbox false = [v 10]
  ∧ mem_node (c 50) escaping_argument_cannot_change = true
  ∧ mem_pair (m 6) (v 10) (t_alias escaping_argument_tables) = true
  ∧ mem_node (m 6) (t_any_source escaping_argument_tables) = true
  ∧ existsb (fun '(x, _) => node_eqb x (m 6))
      (t_sources escaping_argument_tables) = false
  ∧ unit_calling_convention_changes_u escaping_argument_facts
      escaping_argument_tables escaping_argument_cannot_change
      escaping_argument_code_deps escaping_argument_dominated
      (escaping_argument_to_unbox false) escaping_argument_unit
    = [(50, Not_changing_u)]
  ∧ free_variables_of_expr escaping_argument_unit = []
  ∧ free_variables_of_expr (escaping_argument_rebuilt false) = [10]
  ∧ escaping_argument_seeds_of_unit = [v 10]
  ∧ escaping_argument_to_unbox true = []
  ∧ check_field_readers escaping_argument_facts escaping_argument_tables
      escaping_argument_dominated (escaping_argument_to_unbox false) = true
  ∧ check_escaping_arguments mixed_field_facts mixed_field_tables
      mixed_field_cannot_change (mixed_field_to_unbox false) = true.

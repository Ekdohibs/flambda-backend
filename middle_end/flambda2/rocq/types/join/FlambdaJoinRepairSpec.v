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


(* The join with the repaired test for recovering relations: what the
   repair gains, and what it does not.

   This file is to be reviewed: the definitions of the instance
   ([Brought_in]) and the statements. The proofs are in
   FlambdaJoinRepairProofs.v, which needs no review.

   The repair. In a round of the join ([n_way_join_round],
   FlambdaJoinEnv.v, after types/env/join_env.ml) the compiler
   recovers the relations that the type joined for a variable names
   ([recover_inverse_relations]) if every joined environment gave a
   type for the variable. The repaired test
   ([Join_env.repaired_choices], in the same file, beside
   [Join_env.compiler_choices]) asks one thing more before the
   relations are recovered: in every joined environment, what the
   variable stands for (the [Simple] of its definition there, if the
   join made the variable; the variable itself, otherwise) has a
   canonical element at the normal name mode that is bound there at
   that mode, that is, an alias that may occur in terms. Where the
   test answers no, the type joined is the equation of the variable
   as it is, and no relation is recorded. Nothing else of the join
   changes: the first phase, the join of types, the variables brought
   in and the last phase are the compiler's. This is the test of the
   change to join_env.ml that was proposed for discrepancy T5-10 and
   measured against the compiler (the change is not in this tree),
   written as one test where that change has two cases (a variable of
   the first environment that may occur in terms; otherwise, the
   canonical element at each joined environment). It is not shown
   here that the two tests answer alike.

   What the repair gains. The statements about the model's join have
   the value of the target on the runs that have four conditions
   (FlambdaJoinMeetSpec.v, [model_join_scoped_envs_satisfied_statement]
   for the join that the meet calls, [model_join_with_admits_use_statement]
   for the one that Simplify calls). With the repaired test one of the
   four follows from another condition, which is part of what it is
   for a machine to satisfy a typing environment:
   - [relation_scrutinees_have_values] (FlambdaJoinDefineSpec.v):
     where the variable that holds the result of a relation kept has
     a value, the variable that the relation is on has one. With the
     repaired test it holds at the run of a joined environment that
     gives a value to every variable that may occur in terms there
     ([bound_in_terms], FlambdaTypingEnvSpec.v):
     [repaired_scrutinees_have_values_statement]. That condition is
     what [env_satisfies_strictly] (same file) asks where it relates
     the variables of a machine to a typing environment: the machine
     binds the variables that may occur in terms, and the valuation
     has what the machine has for them. That definition is not a
     proof that the runs of the compiler's terms satisfy the
     environments that Simplify builds; nothing here shows that. And
     the condition is not a premise of the contract that the meet has
     for its join, nor of the contract of the join that Simplify
     calls: both are for every valuation that satisfies the
     environments.
   So the two statements, at the repaired test, have that premise in
   the place of the condition ([repaired_join_scoped_envs_satisfied_statement],
   [repaired_join_with_admits_use_statement]: the statements of
   FlambdaJoinMeetFitsSpec.v, where the premise about the types that
   the join adds is discharged, with one premise replaced).

   With nothing supposed of the run, one instance of the transcribed
   join with the repaired test has not the strict contract
   ([sound_join_of_environments], FlambdaJoinSpec.v, which is for
   every valuation that satisfies a use and the source): the
   instance of FlambdaJoinEnvSpec.v ([Witness]) with its variable
   [x] made a variable that may occur in terms, and with the
   stand-ins of [Witness] for the join of types and for the
   environment of the meet ([In_terms],
   [repaired_join_not_strict_statement]). The valuation gives [x]
   no value. This is of that instance and those stand-ins: nothing
   is shown here of the join of types and the meet of the model, nor
   of the compiler's. Such a valuation is not one that
   [env_satisfies_strictly] (FlambdaTypingEnvSpec.v) relates to a
   machine, since that asks [bound_in_terms] of the machine and
   that the valuation agrees with it.

   On the three joins of the shape of the program of discrepancy T5-10
   (FlambdaJoinMeetExamplesSpec.v, [Recovered]), by computation
   ([Three_joins], [three_joins_compiler_statement],
   [three_joins_repaired_statement]): with the repaired test the first
   two joins answer the environments that the compiler's test answers
   (the relation of the first is on a variable that may occur in
   terms, and is kept); the third records no relation on the older
   variable [t0], of which nothing is then known; and the environment
   in which [t0] is 2, which is bottom with the compiler's test, is
   not.

   What stays, and why:
   - [tags_at] (FlambdaJoinStateSpec.v): the blocks of the run have
     tags in range. It is a rule of well-formed terms (decision D23;
     FlambdaJoinSpec.v, [blocks_have_tags], says what is stated of it
     for the machine and what is owed to have it of a valuation). It
     is needed by any join that makes a [Get_tag] relation from the
     variable that a type of variants has for the tag, with either
     test.
   - [aliases_have_values] (FlambdaJoinDefineSpec.v): a variable of
     the target that takes the place of a variable that the join made
     has a value where that one has. It follows from two conditions
     ([aliases_have_values_of_values_statement], same file). One of
     them, that a variable of the first environment that has an
     equation in the level of every joined environment has a value,
     is what the contract of the join that the meet calls supposes of
     a run ([equations_valued], FlambdaMeetSpec.v):
     [equations_valued_joined_have_values_statement]. The other, that
     a variable brought in has a value at a joined environment that
     knows it, is not shown from anything, and no instance is known
     in which it fails on a run of the contract. The repair does not
     bear on it.
   - That every equation added is on a name that has a value. The
     repair does not bear on it, and it cannot go: the instance below
     is a run of the contract of the join that the meet calls on
     which the join, with either test, answers a result that has not
     the conclusion of the contract.

   The instance ([Brought_in]). The first environment knows [p], a
   value, and [g], a naked immediate, and says nothing of them. The
   first joined environment defines [x], for types only, as a value
   that may be null, of which [g] holds whether it is; and [p] is
   there a block of tag 0 whose field is [x]. In the second joined
   environment [p] is the tagged immediate 3 or 4. The join brings
   [x] into its result with the type that it has in the first joined
   environment: the type joined for [p] has a case for the block, and
   the field of that case is [x]. The run: [p] is 3 and [g] is 7. It
   satisfies the first environment and the second joined environment,
   and the one equation that the second has after the scope of the
   first is on [p], which has a value. In the result [x] has an
   equation in the level of the scope of the first environment, which
   has no equation there; so the contract asks that the valuation
   answered give [x] a value ([keeps_equations_valued],
   FlambdaMeetSpec.v), and it is to be a value of the type of [x],
   which no value is where [g] is 7; and [g] is known to the second
   joined environment, so that the valuation answered has 7 for it.
   Hence ([brought_in_no_valuation_statement]) no valuation has the
   three parts of the conclusion, and
   ([model_join_scoped_envs_not_sound_statement]) the join of the
   model has not the contract that the meet has for its join, with
   the compiler's test or the repaired one. No relation on [x] is
   involved: the test for recovering relations is not what fails.
   The part of the conclusion that fails is the one about the names
   of equations: the run, with a value for the one variable that the
   join makes and none for [x], extends the run and satisfies the
   result ([brought_in_result_satisfied_statement]; the reviewer's
   observation). So the result is not an environment that no
   valuation of the run satisfies; what it has not is an equation
   only on names that have values.

   Where the meet needs that part (from its owner): in one lemma,
   which takes a joined environment that is satisfied and whose
   equations after the scope are on names that have values to the
   satisfaction of the extension made of the level cut from it
   ([scoped_extension_satisfied], FlambdaMeetScopesProofs.v); an
   extension is satisfied only if each of its names has a value
   (FlambdaTypeMeaningSpec.v). The extension is then added to the
   environment of the meet, or put in the type answered for one
   alternative of a variant or one case of a row-like type. The
   contract asks every join and every meet to hand that on.

   A way to remove this obstruction, proposed and not proved: it is
   not shown to be enough for the contract, and it is a change of
   another size than the repair. The join would give no equation in
   its result to a variable unless every joined environment has a
   type for what the variable stands for there. That leaves out a
   variable brought in from some of the joined environments only, as
   [x] here, and a variable made whose definition has no [Simple] at
   some joined environment. The type of such a variable would be
   forgotten; or put where the variable is mentioned, as the last
   phase does already for a variable made that is mentioned once; or
   put in the environment extension of the case of the type that
   mentions it, where the meet finds it when the case is known to be
   the one (the meet does that for the variables that it defines
   after a join, its owner says). The test is the compiler's test
   for recovering relations, asked earlier: with it, a name that has
   an equation in the result would be one that has a type in every
   joined environment, and "a name that has an equation has a value"
   might be kept by every join, which is also what the compiler's
   test for relations supposes. None of this is defined here and
   nothing is proved of it. The other ways are not changes of the
   join: the contract of the meet for its join, or the meaning of an
   equation of an environment extension on a name that has no value
   (which the owner of the meet says would make the meet wrong where
   it drops a case whose extensions do not meet).

   Not here: the same question for the contract of the join that
   Simplify calls ([sound_join_of_environments], FlambdaJoinSpec.v),
   which does not ask that the names of equations have values. The
   instance does not refute it; the statements above have its
   conclusion on the runs that have the conditions, and whether it
   holds of the repaired join on every run that has tags in range and
   values for the variables that may occur in terms is open. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaClosuresSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaTypingEnv
  FlambdaTypingEnvSpec FlambdaTypingEnvInvariantSpec
  FlambdaTypingEnvKindedSpec FlambdaJoinEnv FlambdaJoinSpec
  FlambdaJoinEnvSpec FlambdaJoinStateSpec FlambdaJoinDefineSpec
  FlambdaJoinAddsSpec FlambdaJoinTypes FlambdaJoinTypesExamplesSpec
  FlambdaMeetGrammar FlambdaMeetEnv FlambdaMeet FlambdaMeetSpec
  FlambdaMeetExamplesSpec FlambdaJoinMeet FlambdaJoinMeetSpec
  FlambdaJoinMeetExamplesSpec FlambdaJoinMeetFitsSpec.
Import ListNotations.

(* ======================================================================== *)
(* What the repair gains                                                    *)
(* ======================================================================== *)

(* The two loops of rounds of a join, run with the repaired test from
   no relation: at a joined environment, under a valuation that
   satisfies it and gives a value to every variable that may occur in
   terms there, the variable that each relation kept is on has a
   value in the valuation of the target ([target_valuation],
   FlambdaJoinStateSpec.v). The join of types is supposed to keep the
   definitions of the state, as the model's does
   ([n_way_join_of_types_keeps_definitions_statement],
   FlambdaJoinTypesWithEnvSpec.v). *)
Definition repaired_scrutinees_have_values_statement : Prop :=
  ∀ (depth : nat) (p : Join_env.parameters) (fuel : nat)
    (jt : Join_env.n_way_join_type) (s1 s2 s3 : Join_env.t)
    (equations : finite_map (Keyed.finite_map Index.t flambda_type))
    (existentials : finite_map (Simples_in_joined_envs.t * flambda_kind))
    (relations2 relations3 : Join_env.inverse_relations) (index : Index.t)
    (use : Typing_env.t) (statics : static_env) (valuation : env),
    join_of_types_keeps_definitions jt ->
    Join_env.n_way_join_delayed_equations fuel Join_env.repaired_choices jt
      s1 [] equations
    = Some (relations2, s2) ->
    Join_env.n_way_join_loop depth p fuel Join_env.repaired_choices jt s2
      relations2 existentials
    = Some (relations3, s3) ->
    Join_env.joined_env s3 index = Some use ->
    typing_env_satisfied_strictly statics valuation use ->
    bound_in_terms use valuation ->
    relation_scrutinees_have_values s3 relations3 index use valuation.

(* The join that the meet calls, with the repaired test:
   [model_join_scoped_envs_satisfied_outright_statement]
   (FlambdaJoinMeetFitsSpec.v) at [Join_env.repaired_choices], with
   [bound_in_terms] of the joined environment in the place of
   [relation_scrutinees_have_values]. The three other conditions on
   the run are as there. *)
Definition repaired_join_scoped_envs_satisfied_statement : Prop :=
  ∀ (join_depth : nat) (fuel : fuels) (meet_head : meet_expanded_head)
    (initial_env result : Meet_env.t) (scoped_envs : list Typing_env.t)
    (envs : Joined_envs.t) (s1 s2 s3 : Join_env.t)
    (relations2 relations3 : Join_env.inverse_relations)
    (kept : list (variable * flambda_kind))
    (equations relation_equations : list (name * flambda_type))
    (definitions : finite_map (Simples_in_joined_envs.t * flambda_kind))
    (index : Index.t) (scoped_env : Typing_env.t) (statics : static_env)
    (valuation : env),
    let join_choices := Join_env.repaired_choices in
    let initial_tenv := Meet_env.typing_env initial_env in
    let p := join_parameters join_depth in
    let jt := model_join_of_types join_depth fuel in
    let avoid :=
      Meet_env.names_to_avoid initial_env
      ++ Join_env.variables_of_env initial_tenv
      ++ flat_map Join_env.variables_of_env scoped_envs
    in
    let s0 :=
      Join_env.create envs
        (Bindings_in_target_env.from_source_env avoid initial_tenv)
    in
    let menv :=
      Meet_env.with_names_to_avoid initial_env
        (Bindings_in_target_env.names_to_avoid (Join_env.bindings s3))
    in
    meet_expanded_head_sound meet_head ->
    meet_expanded_head_keeps meet_head ->
    typing_env_fits initial_tenv ->
    (∀ scoped_env : Typing_env.t,
      scoped_env ∈ scoped_envs -> scoped_env_of initial_tenv scoped_env) ->
    scoped_envs_agree scoped_envs ->
    model_join_scoped_envs join_choices join_depth fuel meet_head
      (Typing_env.current_scope initial_tenv) initial_env initial_tenv
      scoped_envs
    = Some result ->
    Join_env.cut_for_joins (Typing_env.current_scope initial_tenv)
      (Index.number scoped_envs)
    = Some envs ->
    Join_env.join_aliases_into_bindings p (fuel_of_env fuel) s0 envs
    = Some s1 ->
    Join_env.n_way_join_delayed_equations (fuel_of_env fuel) join_choices jt
      s1 [] (Join_env.new_equations_in_joined_envs s1 s0)
    = Some (relations2, s2) ->
    Join_env.n_way_join_loop join_depth p (fuel_of_env fuel) join_choices jt
      s2 relations2 (Join_env.new_definitions_of_existentials s2 s0)
    = Some (relations3, s3) ->
    last_phase_adds
      (Meet_env.typing_env
         (variables_defined meet_env_functions menv
            (Bindings_in_target_env.imported_variables_in_order
               (Join_env.bindings s3))))
      (fuel_of_env fuel) s3 relations3 kept equations relation_equations
      definitions ->
    Join_env.joined_env s3 index = Some scoped_env ->
    statics_valid statics ->
    typing_env_satisfied_strictly statics valuation initial_tenv ->
    typing_env_satisfied_strictly statics valuation scoped_env ->
    bound_in_terms scoped_env valuation ->
    tags_at s3 scoped_env statics valuation ->
    aliases_have_values s3 index scoped_env valuation ->
    (∀ (n : name) (ty : flambda_type),
       (n, ty) ∈ equations ++ relation_equations ->
       ∃ v : value,
         value_of_name (target_valuation s3 index scoped_env valuation) n
         = Some v) ->
    ∃ valuation' : env,
      extended_from scoped_env valuation valuation'
      ∧ typing_env_satisfied_strictly statics valuation'
          (Meet_env.typing_env result)
      ∧ keeps_equations_valued initial_tenv (Meet_env.typing_env result)
          valuation valuation'.

(* The join that Simplify calls, with the repaired test and any meet
   of expanded types that has the two contracts:
   [model_join_with_admits_use_outright_statement]
   (FlambdaJoinMeetFitsSpec.v) at [Join_env.repaired_choices], with
   [bound_in_terms] of the use in the place of
   [relation_scrutinees_have_values]. *)
Definition repaired_join_with_admits_use_statement : Prop :=
  ∀ (A : Type) (join_depth : nat) (fuel : fuels) (mh : meet_expanded_head)
    (avoid : list variable) (inputs : join_inputs A) (target : Typing_env.t)
    (analysis : Analysis.t A) (envs : Joined_envs.t) (s3 : Join_env.t)
    (relations3 : Join_env.inverse_relations)
    (kept : list (variable * flambda_kind))
    (equations relation_equations : list (name * flambda_type))
    (definitions : finite_map (Simples_in_joined_envs.t * flambda_kind))
    (index : Index.t) (use : Typing_env.t) (statics : static_env)
    (valuation : env),
    let join_choices := Join_env.repaired_choices in
    meet_expanded_head_keeps mh ->
    meet_expanded_head_sound mh ->
    join_inputs_well_formed typing_env_invariant avoid inputs ->
    uses_kinded inputs ->
    uses_agree_with_source_on_kinds inputs ->
    typing_env_fits (source inputs) ->
    uses_fit_otherwise inputs ->
    model_join_with join_choices join_depth fuel mh avoid inputs
    = Some (target, analysis) ->
    model_join_run join_choices join_depth fuel avoid inputs envs s3
      relations3 kept equations relation_equations definitions ->
    use_at inputs index use ->
    Typing_env.is_bottom use = false ->
    statics_valid statics ->
    typing_env_satisfied_strictly statics valuation use ->
    typing_env_satisfied_strictly statics valuation (source inputs) ->
    bound_in_terms use valuation ->
    tags_at s3 use statics valuation ->
    aliases_have_values s3 index use valuation ->
    (∀ (n : name) (ty : flambda_type),
       (n, ty) ∈ equations ++ relation_equations ->
       ∃ v : value,
         value_of_name (target_valuation s3 index use valuation) n = Some v) ->
    ∃ valuation' : env,
      typing_env_satisfied_strictly statics valuation' target
      ∧ agrees_on_known use valuation valuation'
      ∧ agrees_on_known (source inputs) valuation valuation'
      ∧ definitions_hold (Analysis.definitions_in_joined_envs analysis) index
          valuation valuation'
      ∧ definitions_hold
          (Analysis.canonical_definitions_at_normal_mode analysis) index
          valuation valuation'.

(* ======================================================================== *)
(* The strict contract, the repaired test and stand-ins: an instance        *)
(* ======================================================================== *)

(* The instance [Witness] of FlambdaJoinEnvSpec.v, with [x] defined
   as a variable that may occur in terms, and the join with the
   repaired test. The types, the stand-ins and the run are those of
   [Witness]: [r] holds 0 and [x] has no value. *)
Module In_terms.

  Definition source_env : Typing_env.t :=
    Typing_env.add_variable_definition
      (Typing_env.add_variable_definition Typing_env.create Witness.r
         (Naked_number Kind_naked_immediate) Name_mode.normal)
      Witness.x Value Name_mode.normal.

  Definition use_env : Typing_env.t :=
    Typing_env.replace_equation (Typing_env.increment_scope source_env)
      (Name_var Witness.x) Witness.type_of_x.

  Definition use_env_null : Typing_env.t :=
    Typing_env.replace_equation (Typing_env.increment_scope source_env)
      (Name_var Witness.x) Witness.type_of_x_null.

  Definition inputs : join_inputs unit :=
    {| source := source_env;
       cut_after := Typing_env.current_scope source_env;
       uses := [(tt, use_env); (tt, use_env_null)] |}.

  Definition join : join_of_environments unit :=
    FlambdaJoinEnvSpec.repaired_join Witness.ops Witness.parameters 10
      Witness.join_of_types tt.

End In_terms.

(* The inputs are what the contract asks. *)
Definition in_terms_inputs_well_formed_statement : Prop :=
  join_inputs_well_formed typing_env_invariant [Witness.r; Witness.x]
    In_terms.inputs
  ∧ uses_kinded In_terms.inputs.

(* The run satisfies the first use and the source, strictly; and it
   gives no value to [x], which may occur in terms at that use. *)
Definition in_terms_run_statement : Prop :=
  (∀ statics : static_env,
     typing_env_satisfied_strictly statics Witness.valuation
       In_terms.use_env
     ∧ typing_env_satisfied_strictly statics Witness.valuation
         In_terms.source_env)
  ∧ ~ bound_in_terms In_terms.use_env Witness.valuation.

(* The join with the repaired test answers, and its target does not
   admit the first use: the relation on [r] is recovered, since [x]
   may occur in terms, and the run gives [x] no value. *)
Definition in_terms_use_not_admitted_statement : Prop :=
  ∃ (target : Typing_env.t) (analysis : Analysis.t unit),
    In_terms.join [Witness.r; Witness.x] In_terms.inputs
    = Some (target, analysis)
    ∧ ~ join_admits_use In_terms.source_env In_terms.use_env target
          (Analysis.definitions_in_joined_envs analysis)
          (Analysis.canonical_definitions_at_normal_mode analysis) 0.

(* So the join with the repaired test, with these stand-ins, has not
   the strict contract. *)
Definition repaired_join_not_strict_statement : Prop :=
  ~ sound_join_of_environments typing_env_invariant In_terms.join.

(* ======================================================================== *)
(* The three joins of discrepancy T5-10, with either test                   *)
(* ======================================================================== *)

(* The three joins of [Recovered] (FlambdaJoinMeetExamplesSpec.v,
   which says what program they are of), with the test for recovering
   relations as an argument: the same inputs, made in the same way,
   with the same names. The meet that makes the environments calls
   the join with that test too. *)
Module Three_joins.

  Import Recovered.

  Section With_choices.

  Variable choices : Join_env.choices.

  Definition add_equation : add_equation_function :=
    model_add_equation choices compiler_choices 5 example_fuel.

  Definition join (made : list variable) (source : Typing_env.t)
      (uses : list Typing_env.t) : option Typing_env.t :=
    let* '(target, _) :=
      model_join choices compiler_choices 5 example_fuel
        ([n; t0; x; p; s; b0; b1; pa; pb; z; pd] ++ made)
        {| source := source;
           cut_after := Typing_env.current_scope source;
           uses := map (fun use => (tt, use)) uses |}
    in
    Some target.

  Definition with_variable (tenv : Typing_env.t) (var : variable)
      (ty : flambda_type) : option Typing_env.t :=
    add_equation
      (Typing_env.add_variable_definition tenv var (kind ty) Name_mode.normal)
      (Name_var var) ty.

  Definition use_of_kx (tag : Z) (block : variable) : option Typing_env.t :=
    let* tenv :=
      add_equation (Typing_env.increment_scope at_kx) (Name_var t0)
        (this_naked_immediate tag)
    in
    let* tenv := with_variable tenv block (example_join_block tag [alias n]) in
    add_equation tenv (Name_var x) (alias block).

  Definition after_kx : option Typing_env.t :=
    let* use0 := use_of_kx 0 b0 in
    let* use1 := use_of_kx 1 b1 in
    join [] at_kx [use0; use1].

  Definition first_use_of_k0 : option Typing_env.t :=
    let* tenv := after_kx in
    let* tenv := with_variable tenv pa (example_join_block 0 [alias x]) in
    add_equation tenv (Name_var p) (alias pa).

  Definition second_use_of_k0 : option Typing_env.t :=
    let* tenv :=
      with_variable (Typing_env.increment_scope at_k0) pb
        (example_join_block 1 [alias n])
    in
    add_equation tenv (Name_var p) (alias pb).

  Definition after_k0 : option Typing_env.t :=
    let* use0 := first_use_of_k0 in
    let* use1 := second_use_of_k0 in
    join [] at_k0 [use0; use1].

  Definition at_k2 : option Typing_env.t :=
    let* tenv := after_k0 in
    Some (define tenv s Value).

  Definition first_use_of_k2 : option Typing_env.t :=
    let* tenv := at_k2 in
    add_equation (Typing_env.increment_scope tenv) (Name_var s) (alias p).

  Definition second_use_of_k2 : option Typing_env.t :=
    let* tenv := at_k2 in
    let* tenv :=
      add_equation (Typing_env.increment_scope tenv) (Name_var t0)
        (this_naked_immediate 0)
    in
    let* tenv := with_variable tenv z (example_join_block 0 [alias n]) in
    let* tenv := with_variable tenv pd (example_join_block 0 [alias z]) in
    add_equation tenv (Name_var s) (alias pd).

  Definition after_k2 : option Typing_env.t :=
    let* source := at_k2 in
    let* use0 := first_use_of_k2 in
    let* use1 := second_use_of_k2 in
    join [tag_of_p] source [use0; use1].

  (* The arm of the last switch where [t0] is 2. *)
  Definition arm_two : option Typing_env.t :=
    let* tenv := after_k2 in
    add_equation tenv (Name_var t0) (this_naked_immediate 2).

  End With_choices.

End Three_joins.

(* With the compiler's test they are the joins of [Recovered]: the
   environments after each of the three, and the arm, are the same. *)
Definition three_joins_compiler_statement : Prop :=
  Three_joins.after_kx Join_env.compiler_choices = Recovered.after_kx
  ∧ Three_joins.after_k0 Join_env.compiler_choices = Recovered.after_k0
  ∧ Three_joins.after_k2 Join_env.compiler_choices = Recovered.after_k2
  ∧ Three_joins.arm_two Join_env.compiler_choices = Recovered.arm_two.

(* With the repaired test, the first two joins answer the same
   environments. After the third: the field of [s], if its tag is 0,
   is a block whose tag is [t0] (the variable made for the field is
   mentioned once and no relation is on it, so its type is put for
   it); nothing is known of [t0]; and the tag of [s] is held by a
   variable made, as with the compiler's test. The target is not
   bottom, and the environment in which [t0] is 2 after it is not
   bottom. *)
Definition three_joins_repaired_statement : Prop :=
  Three_joins.after_kx Join_env.repaired_choices = Recovered.after_kx
  ∧ Three_joins.after_k0 Join_env.repaired_choices = Recovered.after_k0
  ∧ Recovered.types_of (Three_joins.after_k2 Join_env.repaired_choices)
      [Recovered.s; Recovered.field_of_s; Recovered.t0; Recovered.tag_of_s]
    = Some
        [Some
           (example_join_variant None
              (bottom (Naked_number Kind_naked_immediate))
              (Some Recovered.tag_of_s)
              [(0%Z, [Recovered.block_with_tag_in Recovered.t0 Recovered.n]);
               (1%Z, [Recovered.alias Recovered.n])]);
         None;
         Some (unknown (Naked_number Kind_naked_immediate));
         Some (Recovered.tag_of (Or_unknown.Known [0%Z; 1%Z]) Recovered.s)]
  ∧ option_map Typing_env.is_bottom
      (Three_joins.after_k2 Join_env.repaired_choices)
    = Some false
  ∧ option_map Typing_env.is_bottom
      (Three_joins.arm_two Join_env.repaired_choices)
    = Some false.

(* ======================================================================== *)
(* What the contract's premise on a run gives                               *)
(* ======================================================================== *)

(* The premise on a run of the contract that the meet has for its
   join ([equations_valued] of the joined environment, after the
   scope of the first) gives one of the two conditions from which
   [aliases_have_values] follows
   ([aliases_have_values_of_values_statement],
   FlambdaJoinDefineSpec.v): a variable that has an equation in the
   level kept with every joined environment has a value. It is for
   any state whose joined environments are those cut for the join,
   and for either test. *)
Definition equations_valued_joined_have_values_statement : Prop :=
  ∀ (cut_after : Scope.t) (scoped_envs : list Typing_env.t)
    (s : Join_env.t) (index : Index.t) (scoped_env : Typing_env.t)
    (valuation : env),
    Join_env.cut_for_joins cut_after (Index.number scoped_envs)
    = Some (Join_env.joined_envs s) ->
    Join_env.joined_env s index = Some scoped_env ->
    equations_valued scoped_env cut_after valuation ->
    joined_have_values s valuation.

(* ======================================================================== *)
(* The instance: a variable brought in from one joined environment          *)
(* ======================================================================== *)

Module Brought_in.

  (* [p] and [g] may occur in terms; [x] exists only in types, and only
     in the first joined environment. *)
  Definition p : variable := 1%positive.
  Definition g : variable := 2%positive.
  Definition x : variable := 3%positive.

  (* The first environment: nothing is known of [p], a value, nor of
     [g], a naked immediate. *)
  Definition initial_tenv : Typing_env.t :=
    Typing_env.add_variable_definition
      (Typing_env.add_variable_definition Typing_env.create p Value
         Name_mode.normal)
      g (Naked_number Kind_naked_immediate) Name_mode.normal.

  Definition initial_env : Meet_env.t := Meet_env.create initial_tenv.

  (* A value that may be null, of which [g] holds whether it is. *)
  Definition type_of_x : flambda_type :=
    Type_value
      (Ok (No_alias (Head_of_kind_value Unknown (Maybe_null (Some g))))).

  (* The first joined environment: the first environment, one scope
     further, where [x] is defined with its type and [p] is a block
     of tag 0 whose one field is [x]. *)
  Definition first_env : Typing_env.t :=
    Typing_env.replace_equation
      (Typing_env.replace_equation
         (Typing_env.add_variable_definition
            (Typing_env.increment_scope initial_tenv) x Value
            Name_mode.in_types)
         (Name_var x) type_of_x)
      (Name_var p)
      (immutable_block false 0 (Scannable Value_only) Alloc_mode.Heap
         [alias_type_of Value (Var x)]).

  (* The second: the first environment, one scope further, where [p]
     is the tagged immediate 3 or 4. *)
  Definition second_env : Typing_env.t :=
    Typing_env.replace_equation (Typing_env.increment_scope initial_tenv)
      (Name_var p) (these_tagged_immediates [3%Z; 4%Z]).

  Definition fuel : fuels :=
    {| fuel_of_env := 20; fuel_of_types := 20; fuel_of_meet := 20 |}.

  (* A meet of expanded types that has the two contracts
     ([meet_with_forgetful_join_has_contracts_statement],
     FlambdaJoinMeetSpec.v). *)
  Definition meet_head : meet_expanded_head :=
    Meet.meet_expanded_head forgetful_join repaired_choices 20.

  (* The join that the meet calls, of the two environments, with a
     test for recovering relations and the compiler's depth. *)
  Definition join (choices : Join_env.choices) : option Meet_env.t :=
    model_join_scoped_envs choices 5 fuel meet_head
      (Typing_env.current_scope initial_tenv) initial_env initial_tenv
      [first_env; second_env].

  (* The run, through the second joined environment: [p] is 3, [g] is
     7, and [x] has no value. *)
  Definition valuation : env :=
    [(p, Value_const (Tagged_immediate 3));
     (g, Value_const (Naked_immediate 7))].

  (* The variable that the join makes, for the result of [Is_int] on
     [p]: the first that is none of the three. *)
  Definition made : variable := 4%positive.

  (* The run, with 1 for that variable: [p] is an immediate. *)
  Definition valuation_of_result : env :=
    (made, Value_const (Naked_immediate 1)) :: valuation.

End Brought_in.

(* The inputs have what the contract that the meet has for its join
   supposes ([join_scoped_envs_sound], FlambdaMeetSpec.v): the meet of
   expanded types has its two contracts; a meet may be made in the
   first environment; each joined environment is made from it; and
   no [Simple] is of two kinds in them. *)
Definition brought_in_inputs_statement : Prop :=
  meet_expanded_head_sound Brought_in.meet_head
  ∧ meet_expanded_head_keeps Brought_in.meet_head
  ∧ typing_env_fits Brought_in.initial_tenv
  ∧ (∀ scoped_env : Typing_env.t,
      scoped_env ∈ [Brought_in.first_env; Brought_in.second_env] ->
      scoped_env_of Brought_in.initial_tenv scoped_env)
  ∧ scoped_envs_agree [Brought_in.first_env; Brought_in.second_env].

(* The run has what the contract supposes of a run, at the second
   joined environment and whatever the static environment: it
   satisfies the first environment and that one, and the equations
   that that one has after the scope of the first are on names that
   have a value. And it gives a value to every variable that may
   occur in terms there, which the contract does not ask. *)
Definition brought_in_run_statement : Prop :=
  ∀ statics : static_env,
    typing_env_satisfied_strictly statics Brought_in.valuation
      Brought_in.initial_tenv
    ∧ typing_env_satisfied_strictly statics Brought_in.valuation
        Brought_in.second_env
    ∧ equations_valued Brought_in.second_env
        (Typing_env.current_scope Brought_in.initial_tenv)
        Brought_in.valuation
    ∧ bound_in_terms Brought_in.second_env Brought_in.valuation.

(* The join answers, with either test, and no valuation has the three
   parts of the conclusion of the contract about values, whatever the
   static environment. *)
Definition brought_in_no_valuation_statement : Prop :=
  ∀ choices : Join_env.choices,
    choices = Join_env.compiler_choices
    ∨ choices = Join_env.repaired_choices ->
    ∃ result : Meet_env.t,
      Brought_in.join choices = Some result
      ∧ ∀ (statics : static_env) (valuation' : env),
          ~ (extended_from Brought_in.second_env Brought_in.valuation
               valuation'
             ∧ typing_env_satisfied_strictly statics valuation'
                 (Meet_env.typing_env result)
             ∧ keeps_equations_valued Brought_in.initial_tenv
                 (Meet_env.typing_env result) Brought_in.valuation
                 valuation').

(* The two other parts of the conclusion can be had: the run, with a
   value for the variable that the join makes and none for [x],
   extends the run and satisfies the result, with either test and
   whatever the static environment. What fails is the part about the
   names of equations alone. *)
Definition brought_in_result_satisfied_statement : Prop :=
  ∀ choices : Join_env.choices,
    choices = Join_env.compiler_choices
    ∨ choices = Join_env.repaired_choices ->
    ∃ result : Meet_env.t,
      Brought_in.join choices = Some result
      ∧ extended_from Brought_in.second_env Brought_in.valuation
          Brought_in.valuation_of_result
      ∧ lookup Brought_in.x Brought_in.valuation_of_result = None
      ∧ ∀ statics : static_env,
          typing_env_satisfied_strictly statics
            Brought_in.valuation_of_result (Meet_env.typing_env result).

(* So the join of the model that the meet calls has not the contract
   that the meet has for its join, with the compiler's test or the
   repaired one. *)
Definition model_join_scoped_envs_not_sound_statement : Prop :=
  ∀ choices : Join_env.choices,
    choices = Join_env.compiler_choices
    ∨ choices = Join_env.repaired_choices ->
    ~ join_scoped_envs_sound
        (model_join_scoped_envs choices 5 Brought_in.fuel).

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


(* The join that Simplify calls, on the runs that give a value to
   every variable that a use knows.

   This file is to be reviewed: the definitions and the statements.
   The proofs are in FlambdaJoinInTermsProofs.v, which needs no review.

   What it is for. The contract of the join that Simplify calls
   ([sound_join_of_environments], FlambdaJoinSpec.v) is for every
   valuation that satisfies a use and the source. Two instances of
   the transcribed join, each with stand-ins for the join of types
   and for the environment of the meet, have not that contract
   (FlambdaJoinEnvSpec.v, [Witness], with the compiler's test for
   recovering relations; FlambdaJoinRepairSpec.v, [In_terms], with
   the repaired one). What is merged of its sixth part
   ([join_admits_use]) is for the runs that have four conditions
   (FlambdaJoinMeetFitsSpec.v,
   [model_join_with_admits_use_outright_statement]). Those are
   stated on the state of the join, and not on what a caller has:
   its inputs, the answer and the run. A caller that rewrites the
   handler of a continuation under the join of the environments of
   its applications (FlambdaSimplifyUsesSpec.v,
   [let_cont_handler_joined_statement], which supposes
   [join_guarantees] of an environment) has runs of one sort, and
   can make inputs of one shape. This file states the join for such
   inputs and runs, with premises that are about the inputs, the
   answer and the run only.

   The join. It is the transcribed join with a meet of expanded types
   that is given ([model_join_with], FlambdaJoinMeetSpec.v), which is
   supposed to have the two contracts of a meet, and with any test
   for recovering relations ([Join_env.choices]: the compiler's,
   [Join_env.compiler_choices], and the repaired one,
   [Join_env.repaired_choices], are two). It is not [model_join]: the
   meet of the model is not shown to have the second contract. A meet
   that has both is known
   ([meet_with_forgetful_join_has_contracts_statement], same file).
   The meet is used where the join adds an equation on a name.

   The statements.
   - The five parts of [join_guarantees] that are not about
     valuations, for that join, under premises about the inputs
     only ([model_join_with_guarantees_on_inputs_statement]).
     They are merged for [model_join], at the meet of the model
     ([model_join_guarantees_on_inputs_statement],
     FlambdaJoinMeetFitsSpec.v); this is the same statement with the
     meet given, supposed to have the first contract. The first part
     ([typing_env_fits], FlambdaMeetSpec.v) says that the target has
     the invariant of typing environments, is kinded and has no
     bottom type.
   - The sixth part ([model_join_with_admits_use_valued_statement]),
     for inputs of the shape below, at a use that is not bottom where
     the answer passes a test, and for the runs that have three
     conditions ([run_with_values]): a valuation that satisfies the
     use and the source extends to one that satisfies the target, as
     [join_admits_use] says; and that valuation gives a value to
     every variable that the source knows and to every variable that
     the analysis has a definition for. So a caller whose valuations
     give a value to every known variable has one again for the
     target, where the target knows no other variable (the meets
     that add the equations may define variables; nothing is said of
     what the valuation gives them).
   - The same as [join_admits_use_on] (FlambdaJoinSpec.v), without
     the last part ([model_join_with_admits_use_on_statement]).
   - The same in the machine ([model_join_with_in_machine_statement]):
     if the variables of terms of the source are variables of terms
     at each use, a state of the machine that satisfies the use and
     the source through a valuation that has the conditions satisfies
     the target, through a valuation that has the last part.
   - The test on the answer as a function, and that it gives the
     premise ([kept_defined_at_uses_b_sound_statement]).
   - Where the condition on values comes from, for a use that knows
     only variables that may occur in terms
     ([known_in_terms_have_values_statement]).

   The shape of the inputs, beyond what the merged statements
   suppose of inputs ([uses_know_the_source]): a use knows the
   variables that the source knows, and no other. So no variable is
   brought in from a use.

   The test on the answer ([kept_defined_at]): a variable that the
   analysis has a definition for, and that the source does not know,
   has a [Simple] at the index of the use. Such a variable is one
   that the join has made and kept; its value at a use is that of the
   [Simple] of its definition there, and the proof needs it to have
   one. The last phase of the join keeps a variable that the join
   has made if no variable of the target is to be an alias of it and
   it is mentioned more than once in the types that the join adds or
   is part of a relation (a variable mentioned once is replaced by
   its type). In the last example below the definition of such a
   variable lacks a [Simple] at a use: the types that were joined to
   make it are those of the other uses.

   The conditions on the run ([run_with_values]):
   - the static environment is valid ([statics_valid], which the
     statements about the meet's environment suppose);
   - the valuation gives a value to every variable that the use
     knows ([known_have_values]). What it is for a machine to satisfy
     an environment ([env_satisfies_strictly],
     FlambdaTypingEnvSpec.v) gives this only of the variables that
     may occur in terms ([bound_in_terms]); so it is had there for a
     use all of whose variables may occur in terms
     ([known_in_terms]), and otherwise it is for the caller's notion
     of satisfaction to say, of the variables that have gone out of
     scope and of those that exist only in types (for one that a
     join has kept, the last part of the sixth statement above is
     what the join gives);
   - the blocks that the variables that the use knows hold have tags
     in range ([blocks_have_tags], FlambdaJoinSpec.v: a rule of
     well-formed terms, decision D23).

   The test for recovering relations does not matter to these
   statements, as it does to those of FlambdaJoinRepairSpec.v: on
   these runs a variable that a relation is on has a value whichever
   the test. It is a variable that the source knows; or one that the
   join has made, which a variable of the source is to be an alias
   of, or which the last phase defines, and then its definition has
   a [Simple] at the use by the test on the answer.

   The instances that refute the contracts, and what excludes each
   ([counterexamples_excluded_statement]):
   - [Witness] (FlambdaJoinEnvSpec.v) and [In_terms]
     (FlambdaJoinRepairSpec.v): a use knows the variable [x], and
     the valuation gives it no value ([known_have_values] fails).
   - [Brought_in] (FlambdaJoinRepairSpec.v), an instance of the join
     that the meet calls: one of the two joined environments defines
     a variable that the first environment does not know, so
     [uses_know_the_source] fails of inputs made of them.

   What the join computes on inputs of the shape, by computation
   ([Arms]), with either of the two tests: the uses are the arms of
   a [Switch], each the source one scope further in which the
   scrutinee is the constant of the arm and each parameter is its
   argument.
   - The uses have to say so by equations. [Typing_env.add_alias]
     records that two [Simple]s are equal and gives the name that is
     no longer canonical no equation; the join reads the equations
     of the levels cut from the uses, so of uses made with it alone
     the join says nothing ([arms_without_equations_statement]). The
     meet's environment adds the equation of that name after the
     alias ([Meet_env.replace_concrete_equation], FlambdaMeetEnv.v),
     and so do the uses of the other examples.
   - A block in an arm and an immediate in the other
     ([arms_block_and_immediate_statement]): the parameter is a
     variant; the variable of the variant for the result of [Is_int]
     is the scrutinee, which is 0 at the first use and 1 at the
     second as that result is, so the join keeps no variable; the
     relation is recovered on the scrutinee; and the test on the
     answer passes.
   - Three arms and two parameters whose blocks have the same fields
     at the first two arms and that are immediates at the third
     ([arms_shared_field_statement]): the join keeps a variable for
     the field, mentioned by the types of both parameters, whose
     definition has no [Simple] at the third use; the test on the
     answer fails.
   These are computations of the join; the premises of the
   statements above are not shown of these inputs.

   What this does not cover.
   - A run in which a variable that the use knows has no value. The
     cause is in the proof and not known to be in the join: the
     equations are added through the meet's environment, whose
     statement supposes that each name of an equation has a value
     (FlambdaMeetEnvSpec.v,
     [meet_env_add_env_extension_sound_statement]). For the same
     reason the test on the answer is needed: a variable kept whose
     definition has no [Simple] at the use has no value in that run,
     and an equation is added on it. A possible way to remove both
     is a statement that the meet's environment may be given an
     equation on a variable that it has just defined and that has no
     value. It is not formulated here, and that it would be enough
     is not shown.
   - The range of the valuation that satisfies the target. It is the
     one that the meet's environment answers, and it is not said to
     be in range.
   - A use that is bottom, of which the contract says nothing either.

   Nothing is assumed beyond the premises of each statement. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaClosuresSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaTypingEnv
  FlambdaTypingEnvSpec FlambdaTypingEnvInvariantSpec
  FlambdaTypingEnvKindedSpec FlambdaJoinEnv FlambdaJoinSpec
  FlambdaJoinEnvSpec FlambdaJoinTypesExamplesSpec FlambdaMeetGrammar
  FlambdaMeetEnv FlambdaMeet FlambdaMeetSpec FlambdaMeetExamplesSpec
  FlambdaJoinMeet FlambdaJoinMeetSpec FlambdaJoinMeetExamplesSpec
  FlambdaJoinMeetFitsSpec FlambdaJoinRepairSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* Five parts, with the meet given                                          *)
(* ======================================================================== *)

(* [model_join_guarantees_on_inputs_statement]
   (FlambdaJoinMeetFitsSpec.v), for the join with a meet of expanded
   types that is given and has the first contract of a meet. *)
Definition model_join_with_guarantees_on_inputs_statement : Prop :=
  ∀ (A : Type) (join_choices : Join_env.choices) (join_depth : nat)
    (fuel : fuels) (mh : meet_expanded_head) (avoid : list variable)
    (inputs : join_inputs A) (target : Typing_env.t)
    (analysis : Analysis.t A),
    meet_expanded_head_keeps mh ->
    join_inputs_well_formed typing_env_invariant avoid inputs ->
    uses_kinded inputs ->
    uses_agree_with_source_on_kinds inputs ->
    typing_env_fits (source inputs) ->
    uses_fit_otherwise inputs ->
    model_join_with join_choices join_depth fuel mh avoid inputs
    = Some (target, analysis) ->
    typing_env_fits target
    ∧ join_keeps_source (source inputs) target
    ∧ join_introduces avoid inputs target
    ∧ analysis_describes_uses inputs target analysis.

(* ======================================================================== *)
(* The shape of the inputs, the test on the answer, and the run             *)
(* ======================================================================== *)

(* A use knows the variables that the source knows, and no other. *)
Definition uses_know_the_source {A : Type} (inputs : join_inputs A) : Prop :=
  ∀ (index : Index.t) (use : Typing_env.t) (x : variable),
    use_at inputs index use -> (knows use x <-> knows (source inputs) x).

(* A variable that the analysis has a definition for, and that the
   source does not know, has a [Simple] at the index. *)
Definition kept_defined_at {A : Type} (inputs : join_inputs A)
    (analysis : Analysis.t A) (index : Index.t) : Prop :=
  ∀ (x : variable) (simples : Simples_in_joined_envs.t)
    (kind : flambda_kind),
    lookup x (Analysis.definitions_in_joined_envs analysis)
    = Some (simples, kind) ->
    ~ knows (source inputs) x ->
    ∃ simple : simple, Simples_in_joined_envs.find index simples = Some simple.

(* The same as a function, at every use that is not bottom. *)
Definition kept_defined_at_uses_b {A : Type} (inputs : join_inputs A)
    (analysis : Analysis.t A) : bool :=
  forallb
    (fun definition : variable * (Simples_in_joined_envs.t * flambda_kind) =>
       Typing_env.mem None (source inputs) (Name_var (fst definition))
       || forallb
            (fun use : Index.t * (A * Typing_env.t) =>
               Typing_env.is_bottom (snd (snd use))
               || is_some
                    (Simples_in_joined_envs.find (fst use)
                       (fst (snd definition))))
            (Index.number (uses inputs)))
    (Analysis.definitions_in_joined_envs analysis).

Definition kept_defined_at_uses_b_sound_statement : Prop :=
  ∀ (A : Type) (inputs : join_inputs A) (analysis : Analysis.t A)
    (index : Index.t) (use : Typing_env.t),
    kept_defined_at_uses_b inputs analysis = true ->
    use_at inputs index use ->
    Typing_env.is_bottom use = false ->
    kept_defined_at inputs analysis index.

(* A valuation gives a value to every variable that an environment
   knows. *)
Definition known_have_values (tenv : Typing_env.t) (valuation : env) : Prop :=
  ∀ x : variable, knows tenv x -> ∃ v : value, lookup x valuation = Some v.

(* The run, at a use: the static environment is valid, the valuation
   gives a value to every variable that the use knows, and the blocks
   that those variables hold have tags in range. *)
Definition run_with_values (use : Typing_env.t) (statics : static_env)
    (valuation : env) : Prop :=
  statics_valid statics
  ∧ known_have_values use valuation
  ∧ blocks_have_tags (knows use) statics valuation.

(* Every variable that an environment knows may occur in terms
   there. *)
Definition known_in_terms (tenv : Typing_env.t) : Prop :=
  ∀ x : variable, knows tenv x -> in_scope_in_terms tenv x.

(* Then a valuation that gives a value to the variables that may
   occur in terms gives one to every variable that the environment
   knows. *)
Definition known_in_terms_have_values_statement : Prop :=
  ∀ (tenv : Typing_env.t) (valuation : env),
    known_in_terms tenv ->
    bound_in_terms tenv valuation ->
    known_have_values tenv valuation.

(* ======================================================================== *)
(* The sixth part                                                           *)
(* ======================================================================== *)

(* The join, with any test for recovering relations, on inputs of the
   shape, at a use that is not bottom and at which the variables kept
   are defined, for a run that has the conditions: a valuation that
   satisfies the use and the source extends to one that satisfies the
   target, with the rest of what [join_admits_use] concludes; and
   that one gives a value to every variable that the source knows
   and to every variable that the analysis has a definition for. *)
Definition model_join_with_admits_use_valued_statement : Prop :=
  ∀ (A : Type) (join_choices : Join_env.choices) (join_depth : nat)
    (fuel : fuels) (mh : meet_expanded_head) (avoid : list variable)
    (inputs : join_inputs A) (target : Typing_env.t)
    (analysis : Analysis.t A) (index : Index.t) (use : Typing_env.t)
    (statics : static_env) (valuation : env),
    meet_expanded_head_keeps mh ->
    meet_expanded_head_sound mh ->
    join_inputs_well_formed typing_env_invariant avoid inputs ->
    uses_kinded inputs ->
    uses_agree_with_source_on_kinds inputs ->
    typing_env_fits (source inputs) ->
    uses_fit_otherwise inputs ->
    uses_know_the_source inputs ->
    model_join_with join_choices join_depth fuel mh avoid inputs
    = Some (target, analysis) ->
    use_at inputs index use ->
    Typing_env.is_bottom use = false ->
    kept_defined_at inputs analysis index ->
    typing_env_satisfied_strictly statics valuation use ->
    typing_env_satisfied_strictly statics valuation (source inputs) ->
    run_with_values use statics valuation ->
    ∃ valuation' : env,
      typing_env_satisfied_strictly statics valuation' target
      ∧ agrees_on_known use valuation valuation'
      ∧ agrees_on_known (source inputs) valuation valuation'
      ∧ definitions_hold (Analysis.definitions_in_joined_envs analysis) index
          valuation valuation'
      ∧ definitions_hold
          (Analysis.canonical_definitions_at_normal_mode analysis) index
          valuation valuation'
      ∧ ∀ x : variable,
          knows (source inputs) x
          ∨ is_bound x (Analysis.definitions_in_joined_envs analysis) = true
          ->
          ∃ v : value, lookup x valuation' = Some v.

(* The same as the predicate of the contract, for those runs. *)
Definition model_join_with_admits_use_on_statement : Prop :=
  ∀ (A : Type) (join_choices : Join_env.choices) (join_depth : nat)
    (fuel : fuels) (mh : meet_expanded_head) (avoid : list variable)
    (inputs : join_inputs A) (target : Typing_env.t)
    (analysis : Analysis.t A) (index : Index.t) (use : Typing_env.t),
    meet_expanded_head_keeps mh ->
    meet_expanded_head_sound mh ->
    join_inputs_well_formed typing_env_invariant avoid inputs ->
    uses_kinded inputs ->
    uses_agree_with_source_on_kinds inputs ->
    typing_env_fits (source inputs) ->
    uses_fit_otherwise inputs ->
    uses_know_the_source inputs ->
    model_join_with join_choices join_depth fuel mh avoid inputs
    = Some (target, analysis) ->
    use_at inputs index use ->
    Typing_env.is_bottom use = false ->
    kept_defined_at inputs analysis index ->
    join_admits_use_on (run_with_values use) (source inputs) use target
      (Analysis.definitions_in_joined_envs analysis)
      (Analysis.canonical_definitions_at_normal_mode analysis) index.

(* In the machine: if moreover a variable that may occur in terms in
   the source may occur in terms at each use
   ([terms_in_scope_at_uses], FlambdaJoinSpec.v), a state of the
   machine that satisfies the use and the source through a valuation
   that has the conditions satisfies the target, through a valuation
   that agrees with the first on what the use knows and gives a value
   to every variable that the source knows and to every variable that
   the analysis has a definition for. (The premises on the machine
   and the valuation are [use_and_source_satisfied] of that file with
   the valuation named, and the conditions; the first three parts of
   the conclusion are [env_satisfies_strictly] of the target with
   the valuation named.) *)
Definition model_join_with_in_machine_statement : Prop :=
  ∀ (A : Type) (join_choices : Join_env.choices) (join_depth : nat)
    (fuel : fuels) (mh : meet_expanded_head) (avoid : list variable)
    (inputs : join_inputs A) (target : Typing_env.t)
    (analysis : Analysis.t A) (index : Index.t) (use : Typing_env.t)
    (statics : static_env) (vars valuation : env),
    meet_expanded_head_keeps mh ->
    meet_expanded_head_sound mh ->
    join_inputs_well_formed typing_env_invariant avoid inputs ->
    uses_kinded inputs ->
    uses_agree_with_source_on_kinds inputs ->
    typing_env_fits (source inputs) ->
    uses_fit_otherwise inputs ->
    uses_know_the_source inputs ->
    terms_in_scope_at_uses inputs ->
    model_join_with join_choices join_depth fuel mh avoid inputs
    = Some (target, analysis) ->
    use_at inputs index use ->
    Typing_env.is_bottom use = false ->
    kept_defined_at inputs analysis index ->
    bound_in_terms use vars ->
    typing_env_satisfied_strictly statics valuation use ->
    typing_env_satisfied_strictly statics valuation (source inputs) ->
    agrees_in_terms use valuation vars ->
    run_with_values use statics valuation ->
    bound_in_terms target vars
    ∧ ∃ valuation' : env,
        typing_env_satisfied_strictly statics valuation' target
        ∧ agrees_in_terms target valuation' vars
        ∧ agrees_on_known use valuation valuation'
        ∧ ∀ x : variable,
            knows (source inputs) x
            ∨ is_bound x (Analysis.definitions_in_joined_envs analysis)
              = true
            ->
            ∃ v : value, lookup x valuation' = Some v.

(* ======================================================================== *)
(* The instances that refute the contracts                                  *)
(* ======================================================================== *)

(* What excludes each: in [Witness] and in [In_terms], the use knows
   [x] and the valuation gives it no value; in [Brought_in], a joined
   environment knows a variable that the first environment does
   not. *)
Definition counterexamples_excluded_statement : Prop :=
  ~ known_have_values Witness.use_env Witness.valuation
  ∧ ~ known_have_values In_terms.use_env Witness.valuation
  ∧ (knows Brought_in.first_env Brought_in.x
     ∧ ~ knows Brought_in.initial_tenv Brought_in.x).

(* ======================================================================== *)
(* What the join computes on the arms of a [Switch]                         *)
(* ======================================================================== *)

Module Arms.

  (* The scrutinee, the fields, four blocks and two parameters. *)
  Definition s : variable := 1%positive.
  Definition a1 : variable := 2%positive.
  Definition a2 : variable := 3%positive.
  Definition y1 : variable := 4%positive.
  Definition y2 : variable := 5%positive.
  Definition z1 : variable := 6%positive.
  Definition z2 : variable := 7%positive.
  Definition p : variable := 8%positive.
  Definition q : variable := 9%positive.

  Definition define (tenv : Typing_env.t) (x : variable)
      (kind : flambda_kind) : Typing_env.t :=
    Typing_env.add_variable_definition tenv x kind Name_mode.normal.

  (* A variable that is a block of tag 0 whose one field is [field]. *)
  Definition define_block (tenv : Typing_env.t) (x field : variable)
      : Typing_env.t :=
    Typing_env.replace_equation (define tenv x Value) (Name_var x)
      (example_join_block 0 [alias_type_of Value (Var field)]).

  (* The environment at the [Switch]: [y1] and [z1] are blocks whose
     field is [a1], [y2] and [z2] blocks whose field is [a2]. *)
  Definition at_switch : Typing_env.t :=
    define_block
      (define_block
         (define_block
            (define_block
               (define
                  (define
                     (define Typing_env.create s
                        (Naked_number Kind_naked_immediate))
                     a1 Value)
                  a2 Value)
               y1 a1)
            y2 a2)
         z1 a1)
      z2 a2.

  (* The source: the two parameters are defined. *)
  Definition source_env : Typing_env.t :=
    define (define at_switch p Value) q Value.

  (* A variable made equal to a [Simple], by [Typing_env.add_alias]
     alone, or with the equation of the name that is no longer
     canonical. *)
  Definition alias (with_equation : bool) (kind : flambda_kind)
      (tenv : Typing_env.t) (x : variable) (simple : simple)
      : option Typing_env.t :=
    let* result := Typing_env.add_alias tenv (Var x) simple in
    match result with
    | Ok result =>
      let tenv := Typing_env.add_alias_result_t result in
      Some
        (if with_equation
         then
           Typing_env.replace_equation tenv (Typing_env.demoted_name result)
             (alias_type_of kind (Typing_env.canonical_element result))
         else tenv)
    | Unknown | Bottom => None
    end.

  (* The use of an arm: the source one scope further, where the
     scrutinee is the constant of the arm and the parameters are
     their arguments. *)
  Definition use (with_equations : bool) (discriminant : Z)
      (argument_of_p argument_of_q : simple) : option Typing_env.t :=
    let* tenv :=
      alias with_equations (Naked_number Kind_naked_immediate)
        (Typing_env.increment_scope source_env) s
        (Const (Naked_immediate discriminant))
    in
    let* tenv := alias with_equations Value tenv p argument_of_p in
    alias with_equations Value tenv q argument_of_q.

  Definition fuel : fuels :=
    {| fuel_of_env := 20; fuel_of_types := 20; fuel_of_meet := 20 |}.

  (* A meet of expanded types that has the two contracts. *)
  Definition meet_head : meet_expanded_head :=
    Meet.meet_expanded_head forgetful_join repaired_choices 20.

  (* What is shown of an answer: the types of the parameters and of
     the scrutinee, the definitions, the test on the answer, and
     whether the source knows every variable that the target knows. *)
  Record answer : Type := Answer {
    type_of_p : option flambda_type;
    type_of_q : option flambda_type;
    type_of_s : option flambda_type;
    definitions : finite_map (Simples_in_joined_envs.t * flambda_kind);
    kept_defined : bool;
    source_knows_all : bool
  }.

  (* The join of uses, with a test for recovering relations. *)
  Definition join (choices : Join_env.choices)
      (uses : list (option Typing_env.t)) : option answer :=
    let* uses := example_all uses in
    let inputs :=
      {| source := source_env;
         cut_after := Typing_env.current_scope source_env;
         uses := map (fun use => (tt, use)) uses |}
    in
    let* '(target, analysis) :=
      model_join_with choices 5 fuel meet_head
        [s; a1; a2; y1; y2; z1; z2; p; q] inputs
    in
    Some
      {| type_of_p := Typing_env.find target (Name_var p) None;
         type_of_q := Typing_env.find target (Name_var q) None;
         type_of_s := Typing_env.find target (Name_var s) None;
         definitions := Analysis.definitions_in_joined_envs analysis;
         kept_defined := kept_defined_at_uses_b inputs analysis;
         source_knows_all :=
           forallb
             (fun x => Typing_env.mem None source_env (Name_var x))
             (Join_env.variables_of_env target) |}.

  (* The first variable that the join makes in the last example, and
     the second. *)
  Definition made_for_is_int : variable := 13%positive.
  Definition made_for_field : variable := 14%positive.

  (* The two tests. *)
  Definition tests : list Join_env.choices :=
    [Join_env.compiler_choices; Join_env.repaired_choices].

End Arms.

(* Uses made by [Typing_env.add_alias] alone have no equation after
   the cut: nothing is known of the parameters nor of the scrutinee
   in the target, and nothing is defined. *)
Definition arms_without_equations_statement : Prop :=
  ∀ choices : Join_env.choices,
    choices ∈ Arms.tests ->
    Arms.join choices
      [Arms.use false 0 (Var Arms.y1) (Var Arms.z1);
       Arms.use false 1 (Const (Tagged_immediate 0)) (Var Arms.z1)]
    = Some
        {| Arms.type_of_p := Some (unknown Value);
           Arms.type_of_q := Some (unknown Value);
           Arms.type_of_s :=
             Some (unknown (Naked_number Kind_naked_immediate));
           Arms.definitions := [];
           Arms.kept_defined := true;
           Arms.source_knows_all := true |}.

(* [p] is the block [y1] in the arm of 0 and the immediate 0 in the
   arm of 1; [q] is [z1] in both. In the target [p] is a variant
   whose variable for the result of [Is_int] is the scrutinee, [q] is
   [z1], and the scrutinee is 0 or 1 and the result of [Is_int] on
   [p]. The definitions are of the scrutinee and of [p]; no variable
   is kept. *)
Definition arms_block_and_immediate_statement : Prop :=
  ∀ choices : Join_env.choices,
    choices ∈ Arms.tests ->
    Arms.join choices
      [Arms.use true 0 (Var Arms.y1) (Var Arms.z1);
       Arms.use true 1 (Const (Tagged_immediate 0)) (Var Arms.z1)]
    = Some
        {| Arms.type_of_p :=
             Some
               (example_join_variant (Some Arms.s) (this_naked_immediate 0)
                  None
                  [(0, [alias_type_of Value (Var Arms.a1)])]);
           Arms.type_of_q := Some (alias_type_of Value (Var Arms.z1));
           Arms.type_of_s :=
             Some
               (Type_naked_immediate
                  (Ok
                     (No_alias
                        (Head_of_kind_naked_immediate
                           (Or_unknown.Known [0; 1])
                           [(Relation.Is_int, Name_var Arms.p)]))));
           Arms.definitions :=
             [(Arms.s,
               ([(0%nat, Const (Naked_immediate 0));
                 (1%nat, Const (Naked_immediate 1))],
                Naked_number Kind_naked_immediate));
              (Arms.p,
               ([(0%nat, Var Arms.y1); (1%nat, Const (Tagged_immediate 0))],
                Value))];
           Arms.kept_defined := true;
           Arms.source_knows_all := true |}.

(* Three arms: [p] is [y1], [y2] and the immediate 0; [q] is [z1],
   [z2] and the immediate 0. The join keeps a variable for the field
   of the blocks, which the types of [p] and of [q] both mention, and
   its definition has a [Simple] at the first two uses only: the test
   on the answer fails. (It keeps another for the result of
   [Is_int], which is 0, 0 and 1.) *)
Definition arms_shared_field_statement : Prop :=
  ∀ choices : Join_env.choices,
    choices ∈ Arms.tests ->
    ∃ answer : Arms.answer,
      Arms.join choices
        [Arms.use true 0 (Var Arms.y1) (Var Arms.z1);
         Arms.use true 1 (Var Arms.y2) (Var Arms.z2);
         Arms.use true 2 (Const (Tagged_immediate 0))
           (Const (Tagged_immediate 0))]
      = Some answer
      ∧ Arms.type_of_p answer
        = Some
            (example_join_variant (Some Arms.made_for_is_int)
               (this_naked_immediate 0) None
               [(0, [alias_type_of Value (Var Arms.made_for_field)])])
      ∧ Arms.type_of_q answer = Arms.type_of_p answer
      ∧ lookup Arms.made_for_field (Arms.definitions answer)
        = Some ([(1%nat, Var Arms.a2); (0%nat, Var Arms.a1)], Value)
      ∧ lookup Arms.made_for_is_int (Arms.definitions answer)
        = Some
            ([(2%nat, Const (Naked_immediate 1));
              (0%nat, Const (Naked_immediate 0));
              (1%nat, Const (Naked_immediate 0))],
             Naked_number Kind_naked_immediate)
      ∧ Arms.kept_defined answer = false
      ∧ Arms.source_knows_all answer = false.

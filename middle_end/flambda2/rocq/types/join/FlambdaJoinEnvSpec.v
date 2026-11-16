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


(* The environment side of the n-way join: what is stated of the
   functions of FlambdaJoinEnv.v.

   To be reviewed: every definition and statement of this file. The
   contract of a join is FlambdaJoinSpec.v; this file is about the
   transcription of the compiler's join.

   What is here so far:
   - [compiler_join]: the transcription of the join that Simplify
     calls, as a join of environments in the sense of the contract.
     [repaired_join]: the same with another test for recovering
     relations ([Join_env.repaired_choices]), which is not the
     compiler's.
   - Of the rounds, whatever the test: a relation is recorded only on
     a variable for which the test answered [true]
     ([round_records_where_chosen_statement] and the two that follow).
     Of the repaired test: such a variable has a value at every use,
     in a run that binds the variables of terms
     ([repaired_test_gives_values_statement],
     [repaired_rounds_scrutinees_have_values_statement]). On the
     instance below, the repaired join admits both uses
     ([witness_repaired_join_admits_statement]).
   - The state of a join (the last section): the readings that the
     interface with the join of types asks for, a first invariant
     ([state_valid], about the names of the state), and, of the three
     functions that the join of types calls, the parts of their
     contracts that are not about kinds.
   - Kinds, and the contracts (the section after it): [state_kinded],
     and that those three functions meet their contracts
     ([n_way_join_simples_sound_statement] and the two after it).
   - That those three functions keep the definitions of the state at
     every state, with no premise (the last section).
   - An instance, [Witness], that shows that the transcription has not
     the strict contract ([sound_join_of_environments]): two uses in
     which a variable [x] that exists only in types has a type that
     says that [r] holds whether [x] is null (at the first use [x] may
     be null or not, at the second it is null); a run through the
     first use in which [x] has no value and [r] holds 0. The join puts the
     relation back on [r] ("[r] is the result of [Is_null] on [x]"),
     and no valuation that gives [x] no value satisfies that. The run
     meets neither [scrutinees_have_values] nor
     [joined_variables_have_values], two of the conditions that the
     contract of the compiler's join has
     ([sound_join_where_values_are_read]): [x] is a variable of the
     source that both uses have refined. An instance in which only the
     first fails needs the scrutinee to be a variable that the join
     makes, standing at one use for a variable with no value (the
     shape of the program that the compiler compiles wrongly); it is
     owed, with the model's join of types.

   What the instance is made with, and why it shows something of the
   compiler's function and not of its stand-ins. The join is given
   the functions of the environment of the meet, the meet, and the
   join of types. Those of the model are not under this file yet, so
   the instance has stand-ins, each of which does, on the inputs that
   this instance gives it, what the compiler's function does:
   - The join of types is given, for [x], the two types of the uses.
     The stand-in answers the first, which is their upper bound, if
     every type given is one of the two, and unknown otherwise. It is
     an upper bound ([witness_join_of_types_upper_bound_statement]),
     and what it answers mentions only [r], which the source has. (The
     compiler's function, on these two types, joins the two [is_null]
     variables, which are both [r], into [r], and the two parts that
     are not null, unknown and bottom, into unknown: the same type.)
   - The environment of the meet is the typing environment; an
     equation is added by replacing the type of a name whose type is
     unknown, and the stand-in answers [None] for any other equation.
     The compiler's function meets the two types; the meet of unknown
     with a type that is not an alias and has more than one value is
     that type, and it then replaces the type of the name
     ([Meet_env.replace_concrete_equation]).
   - The equality of types that stands for physical equality is one
     that is right of the two types of [x], which are all that it is
     asked about: they differ, so they are joined and not kept as they
     stand. (The equality of the meaning of types is not under this
     file yet.)
   The type of [x] in the uses is what a join
   brings into an environment for a variable of one of its uses whose
   type was made by reading the same field of an older block that
   another variable was read from (discrepancy T5-10 has the stages).
   That such environments, and such a run, come from a program is not
   shown: the instance is about the contract, and is the reason why
   the contract of the compiler's join has its conditions. *)

From Stdlib Require Import ZArith List Bool PeanoNat.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaEnvAliases
  FlambdaEnvAliasesSpec FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaTypingEnvInvariantSpec FlambdaTypingEnvKindedSpec FlambdaJoinEnv
  FlambdaJoinSpec.
Import ListNotations.

(* ======================================================================== *)
(* The compiler's join, as a join of environments                           *)
(* ======================================================================== *)

(* [Join_env.cut_and_n_way_join_with_analysis], given what the model's
   function needs besides the compiler's arguments, and a test for
   recovering relations. *)
Definition join_with {A meet_env meet_expanded_head : Type}
    (choices : Join_env.choices)
    (ops : Join_env.meet_env_functions meet_env meet_expanded_head)
    (p : Join_env.parameters) (fuel : nat)
    (n_way_join_type : Join_env.n_way_join_type)
    (meet_expanded_head_fn : meet_expanded_head) : join_of_environments A :=
  fun (avoid : list variable) (inputs : join_inputs A) =>
    cut_and_n_way_join_with_analysis ops p fuel choices n_way_join_type
      meet_expanded_head_fn avoid (cut_after inputs) (source inputs)
      (uses inputs).

(* The compiler's join: with the compiler's test. *)
Definition compiler_join {A meet_env meet_expanded_head : Type}
    (ops : Join_env.meet_env_functions meet_env meet_expanded_head)
    (p : Join_env.parameters) (fuel : nat)
    (n_way_join_type : Join_env.n_way_join_type)
    (meet_expanded_head_fn : meet_expanded_head) : join_of_environments A :=
  join_with Join_env.compiler_choices ops p fuel n_way_join_type
    meet_expanded_head_fn.

(* The join with the repaired test, which is not the compiler's. *)
Definition repaired_join {A meet_env meet_expanded_head : Type}
    (ops : Join_env.meet_env_functions meet_env meet_expanded_head)
    (p : Join_env.parameters) (fuel : nat)
    (n_way_join_type : Join_env.n_way_join_type)
    (meet_expanded_head_fn : meet_expanded_head) : join_of_environments A :=
  join_with Join_env.repaired_choices ops p fuel n_way_join_type
    meet_expanded_head_fn.

(* ======================================================================== *)
(* The transcription has not the strict contract                            *)
(* ======================================================================== *)

Module Witness.

  (* [r] may occur in terms; [x] exists only in types. *)
  Definition r : variable := 1%positive.
  Definition x : variable := 2%positive.

  (* A value that may be null, of which [r] holds whether it is. *)
  Definition type_of_x : flambda_type :=
    Type_value
      (Ok (No_alias (Head_of_kind_value Unknown (Maybe_null (Some r))))).

  Definition source_env : Typing_env.t :=
    Typing_env.add_variable_definition
      (Typing_env.add_variable_definition Typing_env.create r
         (Naked_number Kind_naked_immediate) Name_mode.normal)
      x Value Name_mode.in_types.

  (* The same of a value that is null. *)
  Definition type_of_x_null : flambda_type :=
    Type_value
      (Ok (No_alias (Head_of_kind_value Bottom (Maybe_null (Some r))))).

  (* A use: the source, one scope further, where [x] has its type. *)
  Definition use_env : Typing_env.t :=
    Typing_env.replace_equation (Typing_env.increment_scope source_env)
      (Name_var x) type_of_x.

  (* The other use, where [x] is null. *)
  Definition use_env_null : Typing_env.t :=
    Typing_env.replace_equation (Typing_env.increment_scope source_env)
      (Name_var x) type_of_x_null.

  Definition inputs : join_inputs unit :=
    {| source := source_env;
       cut_after := Typing_env.current_scope source_env;
       uses := [(tt, use_env); (tt, use_env_null)] |}.

  (* The stand-in for the join of types. *)
  Definition is_type_of_x (t : flambda_type) : bool :=
    match t with
    | Type_value
        (Ok (No_alias (Head_of_kind_value Unknown (Maybe_null (Some r'))))) =>
      Pos.eqb r' r
    | _ => false
    end.

  Definition is_type_of_x_null (t : flambda_type) : bool :=
    match t with
    | Type_value
        (Ok (No_alias (Head_of_kind_value Bottom (Maybe_null (Some r'))))) =>
      Pos.eqb r' r
    | _ => false
    end.

  Definition join_of_types : Join_env.n_way_join_type :=
    fun (s : Join_env.t) (ts : list (Index.t * flambda_type)) =>
      match ts with
      | [] => None
      | _ :: _ =>
        if forallb
             (fun binding =>
                is_type_of_x (snd binding) || is_type_of_x_null (snd binding))
             ts
        then Some (Or_unknown.Known type_of_x, s)
        else Some (Or_unknown.Unknown, s)
      end.

  (* The stand-in for the equality of types: right of the two types of
     [x]. *)
  Definition type_eqb (t t' : flambda_type) : bool :=
    (is_type_of_x t && is_type_of_x t')
    || (is_type_of_x_null t && is_type_of_x_null t').

  (* The stand-in for the environment of the meet. *)
  Definition replace_unknown (tenv : option Typing_env.t)
      (equation : name * flambda_type) : option Typing_env.t :=
    let* tenv := tenv in
    let* ty := Typing_env.find tenv (fst equation) None in
    match ty with
    | Type_value Unknown | Type_naked_immediate Unknown =>
      Some (Typing_env.replace_equation tenv (fst equation) (snd equation))
    | _ => None
    end.

  Definition ops : Join_env.meet_env_functions Typing_env.t unit :=
    {| Join_env.create_meet_env := fun tenv => tenv;
       Join_env.current_typing_env := fun tenv => tenv;
       Join_env.add_variable_definition :=
         Typing_env.add_variable_definition;
       Join_env.add_env_extension :=
         fun tenv extension _ =>
           match extension with
           | Env_extension equations =>
             List.fold_left replace_unknown equations (Some tenv)
           end;
       Join_env.make_bottom := Typing_env.make_bottom;
       Join_env.final_typing_env := fun tenv => tenv;
       Join_env.names_to_avoid := fun _ => [];
       Join_env.with_names_to_avoid := fun tenv _ => tenv |}.

  (* No variable is made and no constant is expanded in this instance:
     the first and the fourth are never called. *)
  Definition parameters : Join_env.parameters :=
    {| Join_env.fresh := fun _ => 3%positive;
       Join_env.type_eqb := type_eqb;
       Join_env.env_extension_eqb := fun _ _ => false;
       Join_env.expand_const := fun _ => type_of_x;
       Join_env.join_depth := 1 |}.

  Definition join : join_of_environments unit :=
    compiler_join ops parameters 10 join_of_types tt.

  (* The same with the repaired test. *)
  Definition repaired_join : join_of_environments unit :=
    FlambdaJoinEnvSpec.repaired_join ops parameters 10 join_of_types tt.

  (* The run: [r] holds 0 and [x] has no value. *)
  Definition valuation : env := [(r, Value_const (Naked_immediate 0))].

End Witness.

(* The stand-in for the join of types answers an upper bound, and
   leaves the state as it is: a value that has one of the types given,
   under a valuation, has the type answered under any valuation that
   gives [r] the same. *)
Definition witness_join_of_types_upper_bound_statement : Prop :=
  ∀ (s s' : Join_env.t) (ts : list (Index.t * flambda_type))
    (ty : flambda_type) (index : Index.t) (t_index : flambda_type)
    (statics : static_env) (valuation valuation' : env) (v : value),
    Witness.join_of_types s ts = Some (Or_unknown.Known ty, s') ->
    In (index, t_index) ts ->
    lookup Witness.r valuation' = lookup Witness.r valuation ->
    type_satisfied statics valuation v t_index ->
    s' = s ∧ type_satisfied statics valuation' v ty.

(* The inputs are what the contract asks: well formed, with the
   invariant of typing environments, and kinded. *)
Definition witness_inputs_well_formed_statement : Prop :=
  join_inputs_well_formed typing_env_invariant [Witness.r; Witness.x]
    Witness.inputs
  ∧ uses_kinded Witness.inputs.

(* The run satisfies the first use and the source, strictly. *)
Definition witness_run_satisfies_statement : Prop :=
  ∀ statics : static_env,
    typing_env_satisfied_strictly statics Witness.valuation Witness.use_env
    ∧ typing_env_satisfied_strictly statics Witness.valuation
        Witness.source_env.

(* The join answers, and its target does not admit the first use: no
   valuation satisfies the target and gives [r] and [x] what the run
   gives them. The run has not the values that the join reads: [r] has
   a value and [x], on which the target has a relation with [r] for
   result, has none; and [x] is a variable of the source that every
   use has an equation on. *)
Definition witness_use_not_admitted_statement : Prop :=
  ∃ (target : Typing_env.t) (analysis : Analysis.t unit),
    Witness.join [Witness.r; Witness.x] Witness.inputs
    = Some (target, analysis)
    ∧ ~ join_admits_use Witness.source_env Witness.use_env target
          (Analysis.definitions_in_joined_envs analysis)
          (Analysis.canonical_definitions_at_normal_mode analysis) 0
    ∧ ~ scrutinees_have_values (known_at_use Witness.inputs 0) target
          (Analysis.definitions_in_joined_envs analysis) 0 Witness.valuation
    ∧ ~ joined_variables_have_values Witness.inputs Witness.valuation.

(* So the transcription of the compiler's join, with these stand-ins,
   has not the strict contract. *)
Definition transcribed_join_not_strict_statement : Prop :=
  ~ sound_join_of_environments typing_env_invariant Witness.join.

(* ======================================================================== *)
(* Where the rounds record a relation                                       *)
(* ======================================================================== *)

(* A relation is among those kept for the variable that holds its
   result. *)
Definition relation_recorded (relations : Join_env.inverse_relations)
    (result : variable) (relation : Relation.t) (scrutinee : name) : Prop :=
  ∃ recorded : list (Relation.t * name),
    lookup result relations = Some recorded
    ∧ (relation, scrutinee) ∈ recorded.

(* The relations recovered from the type joined for a name are on that
   name. *)
Definition recovered_relations_on_name_statement : Prop :=
  ∀ (relations relations' : Join_env.inverse_relations) (n : name)
    (ty ty' : flambda_type) (result : variable) (relation : Relation.t)
    (scrutinee : name),
    Join_env.recover_inverse_relations relations n ty
    = Some (ty', relations') ->
    relation_recorded relations' result relation scrutinee ->
    relation_recorded relations result relation scrutinee ∨ scrutinee = n.

(* A later state has the joined environments of an earlier one, and
   the definitions that it had: the first two parts of [state_grows]
   (FlambdaJoinSpec.v). *)
Definition keeps_definitions (s s' : Join_env.t) : Prop :=
  (∀ index : Index.t,
     Join_env.joined_env s' index = Join_env.joined_env s index)
  ∧ ∀ (x : variable)
      (definition : Simples_in_joined_envs.t * flambda_kind),
      lookup x (Join_env.definitions_of_existentials s) = Some definition ->
      lookup x (Join_env.definitions_of_existentials s') = Some definition.

(* What is supposed of the join of types in the statements below: at
   every state and for every list of types, the state that it answers
   keeps the definitions. This is more than the contract of the join of
   types gives ([n_way_join_type_sound] has [state_grows] only at a
   valid state, for well-formed and kinded inputs, distinct indices and
   types in scope and kinded), and it is not shown of the model's join
   of types. To use the statements below of the whole join, either
   this is proved of that function, or they are stated again with the
   premises of the calls that the rounds make. *)
Definition join_of_types_keeps_definitions
    (n_way_join_type : Join_env.n_way_join_type) : Prop :=
  ∀ (s s' : Join_env.t) (ts : list (Index.t * flambda_type))
    (joined : Or_unknown.t flambda_type),
    n_way_join_type s ts = Some (joined, s') ->
    keeps_definitions s s'.

(* Every relation of the second map is in the first, or is on a
   variable for which the test answered [true] in some state [s0] that
   keeps the definitions of [s] and whose definitions [s'] keeps.
   (Nothing else is said of [s0]: it is not said to be a state that
   the join went through.) *)
Definition recorded_where_chosen (choices : Join_env.choices)
    (s s' : Join_env.t) (relations relations' : Join_env.inverse_relations)
    : Prop :=
  ∀ (result : variable) (relation : Relation.t) (scrutinee : name),
    relation_recorded relations' result relation scrutinee ->
    relation_recorded relations result relation scrutinee
    ∨ ∃ (x : variable) (types : Keyed.finite_map Index.t flambda_type)
        (s0 : Join_env.t),
        scrutinee = Name_var x
        ∧ keeps_definitions s s0
        ∧ keeps_definitions s0 s'
        ∧ Join_env.recover_relations_on choices s0 x types = true.

(* A round, and each of the two loops of rounds, records a relation
   only where the test answers [true]. *)
Definition round_records_where_chosen_statement : Prop :=
  ∀ (choices : Join_env.choices) (n_way_join_type : Join_env.n_way_join_type)
    (s s' : Join_env.t)
    (equations : finite_map (Keyed.finite_map Index.t flambda_type))
    (relations relations' : Join_env.inverse_relations),
    join_of_types_keeps_definitions n_way_join_type ->
    Join_env.n_way_join_round choices n_way_join_type s equations relations
    = Some (relations', s') ->
    keeps_definitions s s'
    ∧ recorded_where_chosen choices s s' relations relations'.

Definition delayed_equations_record_where_chosen_statement : Prop :=
  ∀ (fuel : nat) (choices : Join_env.choices)
    (n_way_join_type : Join_env.n_way_join_type) (s s' : Join_env.t)
    (equations : finite_map (Keyed.finite_map Index.t flambda_type))
    (relations relations' : Join_env.inverse_relations),
    join_of_types_keeps_definitions n_way_join_type ->
    Join_env.n_way_join_delayed_equations fuel choices n_way_join_type s
      relations equations
    = Some (relations', s') ->
    keeps_definitions s s'
    ∧ recorded_where_chosen choices s s' relations relations'.

Definition loop_records_where_chosen_statement : Prop :=
  ∀ (depth : nat) (p : Join_env.parameters) (fuel : nat)
    (choices : Join_env.choices) (n_way_join_type : Join_env.n_way_join_type)
    (s s' : Join_env.t)
    (existentials : finite_map (Simples_in_joined_envs.t * flambda_kind))
    (relations relations' : Join_env.inverse_relations),
    join_of_types_keeps_definitions n_way_join_type ->
    Join_env.n_way_join_loop depth p fuel choices n_way_join_type s relations
      existentials
    = Some (relations', s') ->
    keeps_definitions s s'
    ∧ recorded_where_chosen choices s s' relations relations'.

(* ======================================================================== *)
(* The repaired test                                                        *)
(* ======================================================================== *)

(* A variable for which the repaired test answers [true] has a value at
   every use, in a valuation that satisfies the use and gives a value
   to the variables that may occur in terms there. (Of the valuation
   only that it respects the aliases of the use is needed, so any
   names may be exempt.) *)
Definition repaired_test_gives_values_statement : Prop :=
  ∀ (s : Join_env.t) (x : variable)
    (types : Keyed.finite_map Index.t flambda_type) (index : Index.t)
    (tenv : Typing_env.t) (exempt : name -> Prop) (statics : static_env)
    (valuation : env),
    Join_env.recover_relations_on Join_env.repaired_choices s x types = true ->
    Join_env.joined_env s index = Some tenv ->
    typing_env_satisfied_but exempt statics valuation tenv ->
    bound_in_terms tenv valuation ->
    has_value_at_use (knows tenv) (Join_env.definitions_of_existentials s)
      index valuation x.

(* So, with the repaired test, every relation that the two loops of
   [cut_and_n_way_join0] record is on a variable that has a value at
   every use, in such a valuation: what [scrutinees_have_values] asks,
   of the relations before they are put in the target. *)
Definition repaired_rounds_scrutinees_have_values_statement : Prop :=
  ∀ (depth : nat) (p : Join_env.parameters) (fuel : nat)
    (n_way_join_type : Join_env.n_way_join_type) (s s1 s2 : Join_env.t)
    (equations : finite_map (Keyed.finite_map Index.t flambda_type))
    (existentials : finite_map (Simples_in_joined_envs.t * flambda_kind))
    (relations1 relations2 : Join_env.inverse_relations) (result : variable)
    (relation : Relation.t) (scrutinee : name) (index : Index.t)
    (tenv : Typing_env.t) (exempt : name -> Prop) (statics : static_env)
    (valuation : env),
    join_of_types_keeps_definitions n_way_join_type ->
    Join_env.n_way_join_delayed_equations fuel Join_env.repaired_choices
      n_way_join_type s [] equations
    = Some (relations1, s1) ->
    Join_env.n_way_join_loop depth p fuel Join_env.repaired_choices
      n_way_join_type s1 relations1 existentials
    = Some (relations2, s2) ->
    relation_recorded relations2 result relation scrutinee ->
    Join_env.joined_env s index = Some tenv ->
    typing_env_satisfied_but exempt statics valuation tenv ->
    bound_in_terms tenv valuation ->
    ∃ x : variable,
      scrutinee = Name_var x
      ∧ has_value_at_use (knows tenv)
          (Join_env.definitions_of_existentials s2) index valuation x.

(* On the instance, the repaired join answers a target that admits
   both uses: the relation on [r] is not recovered, since [x] may not
   occur in terms. *)
Definition witness_repaired_join_admits_statement : Prop :=
  ∃ (target : Typing_env.t) (analysis : Analysis.t unit),
    Witness.repaired_join [Witness.r; Witness.x] Witness.inputs
    = Some (target, analysis)
    ∧ ∀ (index : Index.t) (use : Typing_env.t),
        use_at Witness.inputs index use ->
        join_admits_use Witness.source_env use target
          (Analysis.definitions_in_joined_envs analysis)
          (Analysis.canonical_definitions_at_normal_mode analysis) index.

(* ======================================================================== *)
(* The state of a join: its readings and its invariant                      *)
(* ======================================================================== *)

(* The interface between the two halves of the join (FlambdaJoinSpec.v,
   section [Join_of_types]) is stated for any state with three
   readings and an invariant. Here they are for the state of
   FlambdaJoinEnv.v: the environment of an index is
   [Join_env.joined_env], the definitions are
   [Join_env.definitions_of_existentials], and the kinds of the
   variables of the target are read from the bindings. *)

(* The kind of a variable of the target so far: the reading
   [kind_in_target_env] of FlambdaJoinSpec.v, for the state of
   FlambdaJoinEnv.v. A variable of the source has the kind of its type
   there; a variable that the join has made or brought in, the kind
   recorded with it. *)
Definition kind_in_target_env (s : Join_env.t) (x : variable)
    : option flambda_kind :=
  let bindings := Join_env.bindings s in
  match
    Keyed.lookup name_eqb (Name_var x)
      (Typing_env.names_to_types (Bindings_in_target_env.source_env bindings))
  with
  | Some (ty, _) => Some (kind ty)
  | None =>
    match lookup x (Bindings_in_target_env.existential_variables bindings) with
    | Some kind => Some kind
    | None => lookup x (Bindings_in_target_env.imported_variables bindings)
    end
  end.

(* The names that the variables that the join makes must avoid. *)
Definition names_to_avoid (s : Join_env.t) : list variable :=
  Bindings_in_target_env.names_to_avoid (Join_env.bindings s).

(* A variable that the join has made. *)
Definition is_existential (s : Join_env.t) (x : variable) : Prop :=
  is_bound x
    (Bindings_in_target_env.existential_variables (Join_env.bindings s))
  = true.

(* A variable that the join has brought in from the joined
   environments. *)
Definition is_imported (s : Join_env.t) (x : variable) : Prop :=
  is_bound x (Bindings_in_target_env.imported_variables (Join_env.bindings s))
  = true.

(* The names to avoid include every variable that the source or a
   joined environment knows, and every variable that the join has
   made. *)
Definition names_avoided (s : Join_env.t) : Prop :=
  (∀ x : variable, knows (Join_env.source_env s) x -> x ∈ names_to_avoid s)
  ∧ (∀ (index : Index.t) (use : Typing_env.t) (x : variable),
       Join_env.joined_env s index = Some use ->
       knows use x ->
       x ∈ names_to_avoid s)
  ∧ ∀ x : variable, is_existential s x -> x ∈ names_to_avoid s.

(* No environment knows a variable that the join has made. *)
Definition existentials_new (s : Join_env.t) : Prop :=
  ∀ x : variable,
    is_existential s x ->
    ~ knows (Join_env.source_env s) x
    ∧ ∀ (index : Index.t) (use : Typing_env.t),
        Join_env.joined_env s index = Some use -> ~ knows use x.

(* A variable that has a definition is one that the join has made, at
   the kind of the definition; the variable that stands for a vector
   of [Simple]s has that vector for definition; and a variable that
   the join has made has a definition. *)
Definition existentials_recorded (s : Join_env.t) : Prop :=
  (∀ (x : variable) (simples : Simples_in_joined_envs.t)
     (kind : flambda_kind),
     lookup x (Join_env.definitions_of_existentials s) = Some (simples, kind)
     ->
     lookup x
       (Bindings_in_target_env.existential_variables (Join_env.bindings s))
     = Some kind)
  ∧ (∀ (simples : Simples_in_joined_envs.t) (x : variable),
       Bindings_in_target_env.has_existential_for_these_simples
         (Join_env.bindings s) simples
       = Some x ->
       ∃ (simples' : Simples_in_joined_envs.t) (kind : flambda_kind),
         lookup x (Join_env.definitions_of_existentials s)
         = Some (simples', kind)
         ∧ Simples_in_joined_envs.equal simples simples' = true)
  ∧ ∀ x : variable,
      is_existential s x ->
      is_bound x (Join_env.definitions_of_existentials s) = true.

(* A variable that has been brought in is recorded with a kind; the
   source does not know it, and some joined environment does. *)
Definition imports_recorded (s : Join_env.t) : Prop :=
  (∀ x : variable,
     among x (Join_env.imported_vars s) = true -> is_imported s x)
  ∧ ∀ x : variable,
      is_imported s x ->
      ~ knows (Join_env.source_env s) x
      ∧ ∃ (index : Index.t) (use : Typing_env.t),
          Join_env.joined_env s index = Some use ∧ knows use x.

(* The invariant of the state: [state_valid] of FlambdaJoinSpec.v. It
   says what the names of the state are. What the equations that the
   state keeps mean (those of the target made so far, the types still
   to be joined, the aliases of the variables made) is in
   FlambdaJoinStateSpec.v, with the invariant that has both and the
   functions below shown to keep it. *)
Definition state_valid (s : Join_env.t) : Prop :=
  names_avoided s
  ∧ existentials_new s
  ∧ existentials_recorded s
  ∧ imports_recorded s.

(* The value that the definition of a variable has at an input, under
   a valuation of that input: that of the [Simple] that the definition
   has there, if it has one. *)
Definition value_of_definition (s : Join_env.t) (index : Index.t)
    (valuation : env) (x : variable) : option value :=
  match lookup x (Join_env.definitions_of_existentials s) with
  | Some (simples, _) =>
    match Simples_in_joined_envs.find index simples with
    | Some simple => eval_simple valuation simple
    | None => None
    end
  | None => None
  end.

(* What is asked of the source of new names. *)
Definition fresh_avoids (p : Join_env.parameters) : Prop :=
  ∀ names : list variable, ~ Join_env.fresh p names ∈ names.

(* The reading of kinds answers for the variables that the compiler's
   test finds. *)
Definition kind_in_target_env_exists_statement : Prop :=
  ∀ (s : Join_env.t) (x : variable),
    exists_in_target_env kind_in_target_env s x
    = Bindings_in_target_env.exists_in_target_env (Join_env.bindings s) x.

(* The state that a join starts from is valid, if the names to avoid
   include what the environments know. *)
Definition created_state_valid_statement : Prop :=
  ∀ (avoid : list variable) (source : Typing_env.t) (envs : Joined_envs.t),
    (∀ x : variable, knows source x -> x ∈ avoid) ->
    (∀ (index : Index.t) (use : Typing_env.t) (x : variable),
       Joined_envs.get_nth_joined_env envs index = Some use ->
       knows use x ->
       x ∈ avoid) ->
    state_valid
      (Join_env.create envs
         (Bindings_in_target_env.from_source_env avoid source)).

(* What the interface asks of the invariant. *)
Definition state_valid_has_witnesses_statement : Prop :=
  valid_states_have_witnesses Join_env.joined_env
    Join_env.definitions_of_existentials state_valid.


(* The three functions that the join of types calls, against their
   contracts in FlambdaJoinSpec.v ([import_type_sound],
   [n_way_join_simples_sound], [n_way_join_env_extension_sound]), with
   the readings and the invariant above: each statement below has the
   parts of its contract that are not about kinds. The parts about
   kinds ([state_kinded], and that a symbol is answered only at the
   kind of values) are not here yet. *)

(* [Join_env.import_type]. *)
Definition import_type_valid_statement : Prop :=
  ∀ (p : Join_env.parameters) (fuel : nat) (s s' : Join_env.t)
    (ty : flambda_type),
    fresh_avoids p ->
    state_valid s ->
    Join_env.import_type p fuel s ty = Some s' ->
    state_valid s'
    ∧ state_grows Join_env.joined_env Join_env.definitions_of_existentials
        kind_in_target_env s s'
    ∧ type_in_target kind_in_target_env s' ty.

(* [Join_env.n_way_join_simples]. Of the [Simple]s given it is asked
   here only that each is in the scope of the environment of its
   index; the contract asks that each is of the kind given there,
   which is more. *)
Definition n_way_join_simples_valid_statement : Prop :=
  ∀ (p : Join_env.parameters) (fuel : nat) (s s' : Join_env.t)
    (kind : flambda_kind) (simples : list (Index.t * simple))
    (result : Or_bottom.t simple),
    fresh_avoids p ->
    state_valid s ->
    joined_envs_well_formed typing_env_invariant Join_env.joined_env s ->
    join_args_distinct simples ->
    join_args_in_scope Join_env.joined_env simple_in_scope s simples ->
    Join_env.n_way_join_simples p fuel s kind simples = Some (result, s') ->
    state_valid s'
    ∧ state_grows Join_env.joined_env Join_env.definitions_of_existentials
        kind_in_target_env s s'
    ∧ (result = Or_bottom.Bottom -> simples = [])
    ∧ ∀ joined : simple,
        result = Or_bottom.Ok joined ->
        simple_in_target kind_in_target_env s' joined
        ∧ ∀ (index : Index.t) (simple : simple) (use : Typing_env.t)
            (statics : static_env) (valuation valuation' : env) (v : value),
            In (index, simple) simples ->
            Join_env.joined_env s' index = Some use ->
            typing_env_satisfied_strictly statics valuation use ->
            extends_input Join_env.joined_env
              Join_env.definitions_of_existentials s' index valuation
              valuation' ->
            eval_simple valuation simple = Some v ->
            eval_simple valuation' joined = Some v.


(* What is asked of the equality of environment extensions: if it
   finds the first equal to the second, the second holds wherever the
   first does. (Nothing is asked of the variables that the two
   mention: an equality that passes over the equations that do not
   count may find equal two extensions of which one mentions a
   variable that the other does not.) *)
Definition env_extension_eqb_sound (p : Join_env.parameters) : Prop :=
  ∀ (extension extension' : env_extension) (statics : static_env)
    (valuation : env),
    Join_env.env_extension_eqb p extension extension' = true ->
    env_extension_satisfied statics valuation extension ->
    env_extension_satisfied statics valuation extension'.

(* [Join_env.n_way_join_env_extension]. *)
Definition n_way_join_env_extension_valid_statement : Prop :=
  ∀ (p : Join_env.parameters) (fuel : nat) (s : Join_env.t)
    (extensions : list (Index.t * env_extension))
    (result : Or_bottom.t (env_extension * Join_env.t)),
    fresh_avoids p ->
    env_extension_eqb_sound p ->
    state_valid s ->
    join_args_in_scope Join_env.joined_env env_extension_in_scope s extensions
    ->
    Join_env.n_way_join_env_extension p fuel s extensions = Some result ->
    (result = Or_bottom.Bottom -> extensions = [])
    ∧ ∀ (joined : env_extension) (s' : Join_env.t),
        result = Or_bottom.Ok (joined, s') ->
        state_valid s'
        ∧ state_grows Join_env.joined_env
            Join_env.definitions_of_existentials kind_in_target_env s s'
        ∧ env_extension_in_target kind_in_target_env s' joined
        ∧ ∀ (index : Index.t) (extension : env_extension)
            (use : Typing_env.t) (statics : static_env)
            (valuation valuation' : env),
            In (index, extension) extensions ->
            Join_env.joined_env s' index = Some use ->
            typing_env_satisfied_strictly statics valuation use ->
            extends_input Join_env.joined_env
              Join_env.definitions_of_existentials s' index valuation
              valuation' ->
            env_extension_satisfied statics valuation extension ->
            env_extension_satisfied statics valuation' joined.

(* ======================================================================== *)
(* The state of a join: kinds, and the three contracts                      *)
(* ======================================================================== *)

(* The kinds of the state agree with those of the joined environments:
   [state_kinded] of FlambdaJoinSpec.v. A variable of the source that a
   joined environment has at a kind is of that kind in the source; a
   variable brought in is of the kind recorded in every joined
   environment that knows it; and each simple of a definition is of
   the kind of the definition in its environment. *)
Definition state_kinded (s : Join_env.t) : Prop :=
  (∀ (index : Index.t) (use : Typing_env.t) (x : variable)
     (kind : flambda_kind) (ty : flambda_type)
     (binding_time_and_mode : Binding_time.With_name_mode.t),
     Join_env.joined_env s index = Some use ->
     simple_of_kind use (Var x) kind ->
     Keyed.lookup name_eqb (Name_var x)
       (Typing_env.names_to_types (Join_env.source_env s))
     = Some (ty, binding_time_and_mode) ->
     FlambdaTypeGrammar.kind ty = kind)
  ∧ (∀ (x : variable) (kind : flambda_kind) (index : Index.t)
       (use : Typing_env.t),
       lookup x
         (Bindings_in_target_env.imported_variables (Join_env.bindings s))
       = Some kind ->
       Join_env.joined_env s index = Some use ->
       knows use x ->
       simple_of_kind use (Var x) kind)
  ∧ ∀ (x : variable) (simples : Simples_in_joined_envs.t)
      (kind : flambda_kind) (index : Index.t) (simple : simple)
      (use : Typing_env.t),
      lookup x (Join_env.definitions_of_existentials s)
      = Some (simples, kind) ->
      Simples_in_joined_envs.find index simples = Some simple ->
      Join_env.joined_env s index = Some use ->
      simple_of_kind use simple kind.

(* Every environment joined has a property. *)
Definition joined_envs_are (property : Typing_env.t -> Prop)
    (s : Join_env.t) : Prop :=
  ∀ (index : Index.t) (use : Typing_env.t),
    Join_env.joined_env s index = Some use -> property use.

(* The state that a join starts from is kinded, if a variable of the
   source that a joined environment has is of the source's kind
   there. *)
Definition created_state_kinded_statement : Prop :=
  ∀ (avoid : list variable) (source : Typing_env.t) (envs : Joined_envs.t),
    (∀ (index : Index.t) (use : Typing_env.t) (x : variable)
       (kind : flambda_kind) (ty : flambda_type)
       (binding_time_and_mode : Binding_time.With_name_mode.t),
       Joined_envs.get_nth_joined_env envs index = Some use ->
       simple_of_kind use (Var x) kind ->
       Keyed.lookup name_eqb (Name_var x) (Typing_env.names_to_types source)
       = Some (ty, binding_time_and_mode) ->
       FlambdaTypeGrammar.kind ty = kind) ->
    state_kinded
      (Join_env.create envs
         (Bindings_in_target_env.from_source_env avoid source)).

(* The three functions that the join of types calls meet their
   contracts (FlambdaJoinSpec.v), with the readings, the invariant and
   the kinds of this file, and the invariant of typing environments
   for their well-formedness. Asked of the parameters: a new name is
   none of the names to avoid, and the equality of extensions is
   sound. The statements of the section before are the parts of these
   that are not about kinds, under weaker premises. *)

(* [Join_env.n_way_join_simples]. *)
Definition n_way_join_simples_sound_statement : Prop :=
  ∀ (p : Join_env.parameters) (fuel : nat),
    fresh_avoids p ->
    n_way_join_simples_sound typing_env_invariant Join_env.joined_env
      Join_env.definitions_of_existentials kind_in_target_env state_valid
      state_kinded (Join_env.n_way_join_simples p fuel).

(* [Join_env.import_type]. *)
Definition import_type_sound_statement : Prop :=
  ∀ (p : Join_env.parameters) (fuel : nat),
    fresh_avoids p ->
    import_type_sound typing_env_invariant Join_env.joined_env
      Join_env.definitions_of_existentials kind_in_target_env state_valid
      state_kinded (Join_env.import_type p fuel).

(* [Join_env.n_way_join_env_extension]. *)
Definition n_way_join_env_extension_sound_statement : Prop :=
  ∀ (p : Join_env.parameters) (fuel : nat),
    fresh_avoids p ->
    env_extension_eqb_sound p ->
    n_way_join_env_extension_sound typing_env_invariant Join_env.joined_env
      Join_env.definitions_of_existentials kind_in_target_env state_valid
      state_kinded (Join_env.n_way_join_env_extension p fuel).

(* The same two functions, against the contracts for the names of the
   equations inside what is answered
   ([import_type_equations_kinded] and
   [n_way_join_env_extension_equations_kinded], FlambdaJoinSpec.v):
   the type imported and the extension answered are those given, and
   their variables are variables of the target at the kinds that the
   inputs have them. *)
Definition import_type_equations_kinded_statement : Prop :=
  ∀ (p : Join_env.parameters) (fuel : nat),
    fresh_avoids p ->
    import_type_equations_kinded typing_env_invariant Join_env.joined_env
      kind_in_target_env state_valid state_kinded
      (Join_env.import_type p fuel).

Definition n_way_join_env_extension_equations_kinded_statement : Prop :=
  ∀ (p : Join_env.parameters) (fuel : nat),
    fresh_avoids p ->
    n_way_join_env_extension_equations_kinded typing_env_invariant
      Join_env.joined_env kind_in_target_env state_valid state_kinded
      (Join_env.n_way_join_env_extension p fuel).

(* ======================================================================== *)
(* The definitions are kept at every state                                  *)
(* ======================================================================== *)

(* The three functions that the join of types calls keep the
   definitions of the state, at every state: what a join of types that
   changes the state only through them needs in order to keep them
   too ([join_of_types_keeps_definitions]). *)
Definition n_way_join_simples_keeps_definitions_statement : Prop :=
  ∀ (p : Join_env.parameters) (fuel : nat) (s s' : Join_env.t)
    (kind : flambda_kind) (simples : list (Index.t * simple))
    (result : Or_bottom.t simple),
    Join_env.n_way_join_simples p fuel s kind simples = Some (result, s') ->
    keeps_definitions s s'.

Definition import_type_keeps_definitions_statement : Prop :=
  ∀ (p : Join_env.parameters) (fuel : nat) (s s' : Join_env.t)
    (ty : flambda_type),
    Join_env.import_type p fuel s ty = Some s' -> keeps_definitions s s'.

Definition n_way_join_env_extension_keeps_definitions_statement : Prop :=
  ∀ (p : Join_env.parameters) (fuel : nat) (s s' : Join_env.t)
    (extensions : list (Index.t * env_extension)) (joined : env_extension),
    Join_env.n_way_join_env_extension p fuel s extensions
    = Some (Or_bottom.Ok (joined, s')) ->
    keeps_definitions s s'.

(* ======================================================================== *)
(* Two facts for the others                                                 *)
(* ======================================================================== *)

(* The variables listed for an environment are those that it knows:
   so the join that the meet calls ([Join_env.cut_and_n_way_join])
   makes its variables outside what the environments it is given
   know, as the join that Simplify calls is supposed to be told to
   ([avoids_known], FlambdaJoinSpec.v). *)
Definition variables_of_env_knows_statement : Prop :=
  ∀ (tenv : Typing_env.t) (x : variable),
    x ∈ Join_env.variables_of_env tenv <-> knows tenv x.

(* [Join_env.n_way_join_env_extension] answers one of the extensions
   that it is given, or the empty one. (So what is true of each
   extension given and of the empty one, and does not depend on the
   state, is true of the one answered: the join of types uses it for
   the well-formedness of what it answers.) *)
Definition n_way_join_env_extension_answers_given_statement : Prop :=
  ∀ (p : Join_env.parameters) (fuel : nat) (s s' : Join_env.t)
    (extensions : list (Index.t * env_extension)) (extension : env_extension),
    Join_env.n_way_join_env_extension p fuel s extensions
    = Some (Or_bottom.Ok (extension, s')) ->
    extension = empty_env_extension
    ∨ ∃ index : Index.t, (index, extension) ∈ extensions.

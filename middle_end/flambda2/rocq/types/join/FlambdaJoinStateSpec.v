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


(* What the state of the n-way join means: the invariant that the
   whole join is proved from.

   To be reviewed: every definition and statement of this file. The
   proofs are in FlambdaJoinStateProofs.v.

   FlambdaJoinEnvSpec.v has a first invariant of the state of a join
   ([state_valid]), which is about its names. This file has what the
   equations that the state keeps mean, and the invariant with both
   ([state_true]).

   - One valuation of the target for each valuation of an input
     ([target_valuation]): a variable that the join has made has the
     value of its definition at that input, a variable that the input
     or the source knows has what the valuation of the input gives it,
     and every other variable has no value. It extends the valuation
     of the input, as the interface with the join of types asks of a
     witness ([target_valuation_extends_statement]). A variable that
     the join has brought in from other inputs has no value in it at
     an input that does not know it: nothing is asked of such a
     variable by the equations below, which speak of the value of a
     variable only where it has one.
   - What the state means ([state_holds]). Every equation of the
     target made so far is true at every input: the value that its
     variable has in the valuation of the target, if it has one, has
     the type there ([equation_holds]). The types kept to be joined
     for a variable are types of the environments of their indices,
     and at an input where the variable has a value there is one, and
     the value has it ([types_hold]). A variable that is to be an
     alias of a variable that the join has made has the value of that
     one, where it has a value, and is known to the inputs at which
     that one has a definition ([alias_holds]). And each [Simple] of a
     definition is at the index of a joined environment, in its scope
     ([definitions_in_envs]).
   - The equations are asked of the runs whose blocks have tags in
     range ([tags_at], which is [blocks_have_tags] of FlambdaJoinSpec.v
     for the variables that the input or the source knows): the rounds
     need it where they recover a [Get_tag] relation.
   - The invariant ([state_true]) is [state_valid], [state_holds], and
     that the source of the state is a given one: the join of types is
     known to keep the invariant that it is given, and this is what
     tells that it keeps the source.
   - The three functions that the join of types calls meet their
     contracts of FlambdaJoinSpec.v with this invariant
     ([n_way_join_simples_true_statement] and the two after it), and
     the invariant has the witnesses that the interface asks for
     ([state_true_has_witnesses_statement]). So a join of types that
     is proved from those contracts keeps it.
   - The state that a join starts from has it
     ([created_state_true_statement]), and the first phase of the
     join, which gives each variable of the source its alias or keeps
     its types to be joined, keeps it
     ([join_aliases_true_statement]).

   - What [recover_inverse_relations] does to the type joined for a
     variable and to the relations kept
     ([recover_free_variables_statement] and the two after it): the
     type answered is true of a value of the type given, and a
     relation recorded is true of it, if the blocks of the value have
     tags in range.
   - The rounds ([delayed_equations_true_statement],
     [loop_true_statement]) keep the invariant and the truth of the
     relations kept ([relations_hold]), given a join of types that
     meets its contract of FlambdaJoinSpec.v with the invariant. With
     the first phase, that is the state and the relations that the
     join has when it comes to define the variables of the target
     ([first_phases_true_statement]).

   What is asked of the parameters of the model's functions, besides
   what FlambdaJoinEnvSpec.v asks: the equality of types that stands
   for physical equality is sound for meaning
   ([parameters_type_eqb_sound]), and the type that stands for a
   constant is that of the constant ([expand_const_sound]).

   The definition of the variables of the target
   ([define_or_eliminate_variables_and_add_equations]) is in
   FlambdaJoinDefineSpec.v. Not here yet: the contract of the whole
   join. *)

From Stdlib Require Import ZArith List Bool PeanoNat.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaEnvAliases
  FlambdaEnvAliasesSpec FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaTypingEnvInvariantSpec FlambdaTypingEnvKindedSpec FlambdaJoinEnv
  FlambdaJoinSpec FlambdaJoinEnvSpec.
Import ListNotations.

(* ======================================================================== *)
(* One valuation of the target for each valuation of an input               *)
(* ======================================================================== *)

(* A variable that an input, or the source, knows. *)
Definition known_to (s : Join_env.t) (use : Typing_env.t) (x : variable)
    : bool :=
  Typing_env.mem None use (Name_var x)
  || Typing_env.mem None (Join_env.source_env s) (Name_var x).

(* The valuation of the target that a valuation of an input gives: a
   variable that the join has made has the value of its definition at
   that input, if it has one; a variable that the input or the source
   knows has what the valuation gives it; every other variable has no
   value, whatever the valuation gives it. *)
Definition target_valuation (s : Join_env.t) (index : Index.t)
    (use : Typing_env.t) (valuation : env) : env :=
  tabulate (value_of_definition s index valuation)
    (domain (Join_env.definitions_of_existentials s))
  ++ filter_bindings (fun x _ => known_to s use x) valuation.

(* It is a witness of [valid_states_have_witnesses], and it gives the
   variables that the source knows what they had. *)
Definition target_valuation_extends_statement : Prop :=
  ∀ (s : Join_env.t) (index : Index.t) (use : Typing_env.t) (valuation : env),
    state_valid s ->
    Join_env.joined_env s index = Some use ->
    extends_input Join_env.joined_env Join_env.definitions_of_existentials s
      index valuation (target_valuation s index use valuation)
    ∧ agrees_on_known (Join_env.source_env s) valuation
        (target_valuation s index use valuation).

(* ======================================================================== *)
(* What the state means                                                     *)
(* ======================================================================== *)

(* A variable that the target has so far, or that a joined environment
   knows. *)
Definition in_reach (s : Join_env.t) (x : variable) : Prop :=
  exists_in_target_env kind_in_target_env s x = true
  ∨ ∃ (index : Index.t) (use : Typing_env.t),
      Join_env.joined_env s index = Some use ∧ knows use x.

(* The blocks of a run have tags in range, for the variables that the
   input or the source knows. *)
Definition tags_at (s : Join_env.t) (use : Typing_env.t)
    (statics : static_env) (valuation : env) : Prop :=
  blocks_have_tags (fun x => known_to s use x = true) statics valuation.

(* An equation of the target is true at every input: the variable is
   one of the target, the type mentions variables in reach, and the
   value that the variable has in the valuation of the target has the
   type there. (The variables of the type need not be variables of the
   target yet: the join adds an equation before it brings in the
   variables of its type.) *)
Definition equation_holds (s : Join_env.t) (x : variable)
    (ty : flambda_type) : Prop :=
  exists_in_target_env kind_in_target_env s x = true
  ∧ (∀ y : variable, y ∈ free_variables ty -> in_reach s y)
  ∧ ∀ (index : Index.t) (use : Typing_env.t) (statics : static_env)
      (valuation : env) (v : value),
      Join_env.joined_env s index = Some use ->
      typing_env_satisfied_strictly statics valuation use ->
      tags_at s use statics valuation ->
      lookup x (target_valuation s index use valuation) = Some v ->
      type_satisfied statics (target_valuation s index use valuation) v ty.

Definition equations_hold (s : Join_env.t) : Prop :=
  ∀ (x : variable) (ty : flambda_type),
    lookup x (Join_env.types_in_target_env s) = Some ty ->
    equation_holds s x ty.

(* Types kept to be joined for a variable of the target are true of
   it: each is a type of the environment of its index, in scope and
   kinded there; and at an input where the variable has a value, there
   is a type for that input, and the value has it. *)
Definition types_hold (s : Join_env.t) (x : variable)
    (types : Keyed.finite_map Index.t flambda_type) : Prop :=
  exists_in_target_env kind_in_target_env s x = true
  ∧ (∀ (index : Index.t) (t : flambda_type),
       Keyed.lookup Index.equal index types = Some t ->
       ∃ use : Typing_env.t,
         Join_env.joined_env s index = Some use
         ∧ type_in_scope use t
         ∧ type_kinded use t)
  ∧ ∀ (index : Index.t) (use : Typing_env.t) (statics : static_env)
      (valuation : env) (v : value),
      Join_env.joined_env s index = Some use ->
      typing_env_satisfied_strictly statics valuation use ->
      lookup x (target_valuation s index use valuation) = Some v ->
      ∃ t : flambda_type,
        Keyed.lookup Index.equal index types = Some t
        ∧ type_satisfied statics valuation v t.

Definition types_to_join_hold (s : Join_env.t) : Prop :=
  ∀ (x : variable) (types : Keyed.finite_map Index.t flambda_type),
    lookup x (Join_env.types_in_joined_envs s) = Some types ->
    types_hold s x types.

(* Where a variable that is to be an alias of a variable that the join
   has made is: every input at which the definition of that one has a
   [Simple] knows it. (The vector of the definition is made of the
   types that the inputs that know the variable have for it.) *)
Definition alias_located (s : Join_env.t) (e x : variable) : Prop :=
  ∀ (simples : Simples_in_joined_envs.t) (kind : flambda_kind)
    (index : Index.t) (simple : simple),
    lookup e (Join_env.definitions_of_existentials s) = Some (simples, kind)
    ->
    Simples_in_joined_envs.find index simples = Some simple ->
    ∃ use : Typing_env.t,
      Join_env.joined_env s index = Some use ∧ knows use x.

(* A variable has an equation in the level kept with every joined
   environment. *)
Definition joined_everywhere (s : Join_env.t) (x : variable) : Prop :=
  ∀ (index : Index.t) (use : Typing_env.t)
    (equations : Keyed.finite_map name flambda_type),
    Keyed.lookup Index.equal index (Join_env.joined_envs s)
    = Some (use, equations) ->
    Keyed.is_bound name_eqb (Name_var x) equations = true.

(* A variable of the target that is to be an alias of a variable that
   the join has made has the value of that variable, at every input
   where it has a value. And two facts about where it comes from,
   which say nothing of values: it is located ([alias_located]); and
   if the source knows it, it has an equation in every level
   ([joined_everywhere]: only the first phase of the join makes an
   alias of a variable of the source, and it looks at the variables
   that have one). With them, the converse of the first part follows
   from two of the conditions on the runs of the compiler's join
   ([aliases_have_values_of_values_statement],
   FlambdaJoinDefineSpec.v). *)
Definition alias_holds (s : Join_env.t) (e x : variable) : Prop :=
  exists_in_target_env kind_in_target_env s x = true
  ∧ is_existential s e
  ∧ (∀ (index : Index.t) (use : Typing_env.t) (statics : static_env)
       (valuation : env) (v : value),
       Join_env.joined_env s index = Some use ->
       typing_env_satisfied_strictly statics valuation use ->
       lookup x (target_valuation s index use valuation) = Some v ->
       lookup e (target_valuation s index use valuation) = Some v)
  ∧ alias_located s e x
  ∧ (knows (Join_env.source_env s) x -> joined_everywhere s x).

Definition aliases_hold (s : Join_env.t) : Prop :=
  ∀ e x : variable,
    x
    ∈ Aliases_of_existentials.aliases_of_existential_var
        (Join_env.aliases_of_existentials s) e ->
    alias_holds s e x.

(* A vector of [Simple]s is over the joined environments: each of its
   [Simple]s is at the index of one of them, and in its scope. *)
Definition vector_in_envs (s : Join_env.t)
    (simples : Simples_in_joined_envs.t) : Prop :=
  ∀ (index : Index.t) (simple : simple),
    Simples_in_joined_envs.find index simples = Some simple ->
    ∃ use : Typing_env.t,
      Join_env.joined_env s index = Some use ∧ simple_in_scope use simple.

Definition definitions_in_envs (s : Join_env.t) : Prop :=
  ∀ (x : variable) (simples : Simples_in_joined_envs.t)
    (kind : flambda_kind),
    lookup x (Join_env.definitions_of_existentials s) = Some (simples, kind) ->
    vector_in_envs s simples.

(* What the state means: the four together. *)
Definition state_holds (s : Join_env.t) : Prop :=
  equations_hold s
  ∧ types_to_join_hold s
  ∧ aliases_hold s
  ∧ definitions_in_envs s.

(* The invariant of the state of a join whose source is [source]: its
   names, what it means, and that its source is that one. *)
Definition state_true (source : Typing_env.t) (s : Join_env.t) : Prop :=
  state_valid s ∧ state_holds s ∧ Join_env.source_env s = source.

(* ======================================================================== *)
(* The invariant, and the functions that the join of types calls            *)
(* ======================================================================== *)

(* What is asked of the equality of types: if it finds the first equal
   to the second, a value of the second has the first. (The join asks
   it with the type of the least index first, and that is the type
   that it keeps.) *)
Definition parameters_type_eqb_sound (p : Join_env.parameters) : Prop :=
  ∀ (t t' : flambda_type) (statics : static_env) (valuation : env)
    (v : value),
    Join_env.type_eqb p t t' = true ->
    type_satisfied statics valuation v t' ->
    type_satisfied statics valuation v t.

(* The state that a join starts from has the invariant, if the names
   to avoid include what the environments know. *)
Definition created_state_true_statement : Prop :=
  ∀ (avoid : list variable) (source : Typing_env.t) (envs : Joined_envs.t),
    (∀ x : variable, knows source x -> x ∈ avoid) ->
    (∀ (index : Index.t) (use : Typing_env.t) (x : variable),
       Joined_envs.get_nth_joined_env envs index = Some use ->
       knows use x ->
       x ∈ avoid) ->
    state_true source
      (Join_env.create envs
         (Bindings_in_target_env.from_source_env avoid source)).

(* What the interface asks of the invariant. *)
Definition state_true_has_witnesses_statement : Prop :=
  ∀ source : Typing_env.t,
    valid_states_have_witnesses Join_env.joined_env
      Join_env.definitions_of_existentials (state_true source).

(* The three functions that the join of types calls meet their
   contracts (FlambdaJoinSpec.v) with this invariant. *)
Definition n_way_join_simples_true_statement : Prop :=
  ∀ (p : Join_env.parameters) (fuel : nat) (source : Typing_env.t),
    fresh_avoids p ->
    parameters_type_eqb_sound p ->
    n_way_join_simples_sound typing_env_invariant Join_env.joined_env
      Join_env.definitions_of_existentials kind_in_target_env
      (state_true source) state_kinded (Join_env.n_way_join_simples p fuel).

Definition import_type_true_statement : Prop :=
  ∀ (p : Join_env.parameters) (fuel : nat) (source : Typing_env.t),
    fresh_avoids p ->
    parameters_type_eqb_sound p ->
    import_type_sound typing_env_invariant Join_env.joined_env
      Join_env.definitions_of_existentials kind_in_target_env
      (state_true source) state_kinded (Join_env.import_type p fuel).

Definition n_way_join_env_extension_true_statement : Prop :=
  ∀ (p : Join_env.parameters) (fuel : nat) (source : Typing_env.t),
    fresh_avoids p ->
    parameters_type_eqb_sound p ->
    env_extension_eqb_sound p ->
    n_way_join_env_extension_sound typing_env_invariant Join_env.joined_env
      Join_env.definitions_of_existentials kind_in_target_env
      (state_true source) state_kinded
      (Join_env.n_way_join_env_extension p fuel).

(* ======================================================================== *)
(* The aliases of the variables of the source                               *)
(* ======================================================================== *)

(* The equations of the level kept with each joined environment are
   types that the environment has. (The levels are cut from the
   environments: [cut_equations_found_statement],
   FlambdaTypingEnvInvariantSpec.v.) *)
Definition levels_found (envs : Joined_envs.t) : Prop :=
  ∀ (index : Index.t) (use : Typing_env.t)
    (equations : Keyed.finite_map name flambda_type) (n : name)
    (ty : flambda_type),
    Keyed.lookup Index.equal index envs = Some (use, equations) ->
    Keyed.lookup name_eqb n equations = Some ty ->
    ∃ binding_time_and_mode : Binding_time.With_name_mode.t,
      Keyed.lookup name_eqb n (Typing_env.names_to_types use)
      = Some (ty, binding_time_and_mode).

(* [join_aliases_into_bindings] keeps the invariant, from a state whose
   variables are those of the source (the state that the join starts
   from is one), if the state is kinded and the joined environments
   have their invariant and are kinded. *)
Definition join_aliases_true_statement : Prop :=
  ∀ (p : Join_env.parameters) (fuel : nat) (source : Typing_env.t)
    (s s' : Join_env.t),
    fresh_avoids p ->
    parameters_type_eqb_sound p ->
    state_true source s ->
    state_kinded s ->
    joined_envs_are typing_env_invariant s ->
    joined_envs_are typing_env_kinded s ->
    levels_found (Join_env.joined_envs s) ->
    (∀ x : variable,
       exists_in_target_env kind_in_target_env s x = true -> knows source x)
    ->
    Join_env.join_aliases_into_bindings p fuel s (Join_env.joined_envs s)
    = Some s' ->
    state_true source s'
    ∧ state_grows Join_env.joined_env Join_env.definitions_of_existentials
        kind_in_target_env s s'.

(* ======================================================================== *)
(* The relations that the rounds recover                                    *)
(* ======================================================================== *)

(* A relation kept for the variable that holds its result is true at
   every input where the variable that it is on has a value: the
   result is there the number that the relation gives for that value.
   (Where the variable that it is on has no value, nothing is said:
   that the result has no value there either is the condition
   [scrutinees_have_values] on the runs of the compiler's join.) *)
Definition relation_holds_at (s : Join_env.t) (r : variable)
    (relation : Relation.t) (x : variable) : Prop :=
  exists_in_target_env kind_in_target_env s x = true
  ∧ exists_in_target_env kind_in_target_env s r = true
  ∧ ∀ (index : Index.t) (use : Typing_env.t) (statics : static_env)
      (valuation : env) (v : value),
      Join_env.joined_env s index = Some use ->
      typing_env_satisfied_strictly statics valuation use ->
      tags_at s use statics valuation ->
      lookup x (target_valuation s index use valuation) = Some v ->
      ∃ i : Z,
        lookup r (target_valuation s index use valuation)
        = Some (Value_const (Naked_immediate i))
        ∧ relation_holds statics relation v i.

Definition relations_hold (s : Join_env.t)
    (relations : Join_env.inverse_relations) : Prop :=
  ∀ (r : variable) (relation : Relation.t) (n : name),
    relation_recorded relations r relation n ->
    ∃ x : variable, n = Name_var x ∧ relation_holds_at s r relation x.

(* [recover_inverse_relations], given the type joined for a variable
   and the relations kept so far. The type that it answers mentions no
   variable that the type given does not, but the one that the
   relations are on. *)
Definition recover_free_variables_statement : Prop :=
  ∀ (relations relations' : Join_env.inverse_relations) (x : variable)
    (ty ty' : flambda_type) (y : variable),
    Join_env.recover_inverse_relations relations (Name_var x) ty
    = Some (ty', relations') ->
    y ∈ free_variables ty' ->
    y ∈ free_variables ty ∨ y = x.

(* A relation that it records is on that variable; the variable that
   holds its result is one of the type; and of a value of the type
   whose blocks have tags in range, that variable holds the number
   that the relation gives. (A type of variants says of a block of any
   tag that its variable for the tag holds the tag; [Get_tag] relates
   a block to its tag only if the tag is in range.) *)
Definition recovered_relations_hold_statement : Prop :=
  ∀ (relations relations' : Join_env.inverse_relations) (x : variable)
    (ty ty' : flambda_type) (r : variable) (relation : Relation.t) (n : name),
    Join_env.recover_inverse_relations relations (Name_var x) ty
    = Some (ty', relations') ->
    relation_recorded relations' r relation n ->
    relation_recorded relations r relation n
    ∨ (n = Name_var x
       ∧ r ∈ free_variables ty
       ∧ ∀ (statics : static_env) (vars : env) (v : value),
           (∀ (tag : Z) (fields : list value),
              block_of statics v = Some (tag, fields) -> is_tag tag) ->
           type_satisfied statics vars v ty ->
           ∃ i : Z,
             lookup r vars = Some (Value_const (Naked_immediate i))
             ∧ relation_holds statics relation v i).

(* And the value of the variable, if it has the type given and its
   blocks have tags in range, has the type answered: the relation that
   is put in the extension for blocks of a type of variants is true of
   the variable that holds the tag. *)
Definition recover_keeps_meaning_statement : Prop :=
  ∀ (relations relations' : Join_env.inverse_relations) (x : variable)
    (ty ty' : flambda_type) (statics : static_env) (vars : env) (v : value),
    Join_env.recover_inverse_relations relations (Name_var x) ty
    = Some (ty', relations') ->
    lookup x vars = Some v ->
    (∀ (tag : Z) (fields : list value),
       block_of statics v = Some (tag, fields) -> is_tag tag) ->
    type_satisfied statics vars v ty ->
    type_satisfied statics vars v ty'.

(* ======================================================================== *)
(* The rounds                                                               *)
(* ======================================================================== *)

(* What is asked of the type that stands for a constant that is not an
   alias: it mentions no variable, it is kinded in every environment,
   and the constant has it. (The compiler's type for a tagged immediate
   is a type of variants whose type of immediates is an alias of a
   constant: it has an alias inside it, of a constant, which is of its
   kind whatever the environment.) *)
Definition expand_const_sound (p : Join_env.parameters) : Prop :=
  ∀ c : const,
    free_variables (Join_env.expand_const p c) = []
    ∧ (∀ tenv : Typing_env.t, type_kinded tenv (Join_env.expand_const p c))
    ∧ ∀ (statics : static_env) (valuation : env),
        type_satisfied statics valuation (Value_const c)
          (Join_env.expand_const p c).

(* The rounds that join the types kept ([n_way_join_delayed_equations]),
   given a join of types that meets its contract with the invariant:
   from a state that has the invariant, with relations that are true
   and types to join that are true, they come to a state that has the
   invariant, with relations that are true. Whether the relations are
   recovered or not makes no difference here: a relation recovered is
   true where its variable has a value. *)
Definition delayed_equations_true_statement : Prop :=
  ∀ (source : Typing_env.t) (jt : Join_env.n_way_join_type)
    (choices : Join_env.choices) (fuel : nat) (s s' : Join_env.t)
    (equations : finite_map (Keyed.finite_map Index.t flambda_type))
    (relations relations' : Join_env.inverse_relations),
    n_way_join_type_sound typing_env_invariant Join_env.joined_env
      Join_env.definitions_of_existentials kind_in_target_env
      (state_true source) state_kinded jt ->
    (∀ (var : variable) (types : Keyed.finite_map Index.t flambda_type),
       lookup var equations = Some types -> types_hold s var types) ->
    state_true source s ->
    joined_envs_are typing_env_invariant s ->
    joined_envs_are typing_env_kinded s ->
    relations_hold s relations ->
    Join_env.n_way_join_delayed_equations fuel choices jt s relations
      equations
    = Some (relations', s') ->
    state_true source s'
    ∧ relations_hold s' relations'
    ∧ state_grows Join_env.joined_env Join_env.definitions_of_existentials
        kind_in_target_env s s'.

(* The loop over the variables that the join has made
   ([n_way_join_loop]), given definitions that the state has. *)
Definition loop_true_statement : Prop :=
  ∀ (source : Typing_env.t) (jt : Join_env.n_way_join_type)
    (choices : Join_env.choices) (p : Join_env.parameters)
    (depth fuel : nat) (s s' : Join_env.t)
    (relations relations' : Join_env.inverse_relations)
    (existentials : finite_map (Simples_in_joined_envs.t * flambda_kind)),
    n_way_join_type_sound typing_env_invariant Join_env.joined_env
      Join_env.definitions_of_existentials kind_in_target_env
      (state_true source) state_kinded jt ->
    expand_const_sound p ->
    (∀ (x : variable) (definition : Simples_in_joined_envs.t * flambda_kind),
       lookup x existentials = Some definition ->
       lookup x (Join_env.definitions_of_existentials s) = Some definition) ->
    state_true source s ->
    joined_envs_are typing_env_invariant s ->
    joined_envs_are typing_env_kinded s ->
    relations_hold s relations ->
    Join_env.n_way_join_loop depth p fuel choices jt s relations existentials
    = Some (relations', s') ->
    state_true source s'
    ∧ relations_hold s' relations'
    ∧ state_grows Join_env.joined_env Join_env.definitions_of_existentials
        kind_in_target_env s s'.

(* The phases of [cut_and_n_way_join0] before the variables of the
   target are defined, together: the state and the relations that
   [define_or_eliminate_variables_and_add_equations] is given. Asked of
   the joined environments: the names to avoid include what they and
   the source know; each has its invariant and is kinded; the levels
   kept with them are types that they have; and a variable of the
   source that one of them has is of the source's kind there. *)
Definition first_phases_true_statement : Prop :=
  ∀ (source : Typing_env.t) (jt : Join_env.n_way_join_type)
    (choices : Join_env.choices) (p : Join_env.parameters) (fuel : nat)
    (avoid : list variable) (envs : Joined_envs.t) (s1 s2 s3 : Join_env.t)
    (relations2 relations3 : Join_env.inverse_relations),
    n_way_join_type_sound typing_env_invariant Join_env.joined_env
      Join_env.definitions_of_existentials kind_in_target_env
      (state_true source) state_kinded jt ->
    fresh_avoids p ->
    parameters_type_eqb_sound p ->
    expand_const_sound p ->
    (∀ x : variable, knows source x -> x ∈ avoid) ->
    (∀ (index : Index.t) (use : Typing_env.t) (x : variable),
       Joined_envs.get_nth_joined_env envs index = Some use ->
       knows use x ->
       x ∈ avoid) ->
    (∀ (index : Index.t) (use : Typing_env.t),
       Joined_envs.get_nth_joined_env envs index = Some use ->
       typing_env_invariant use) ->
    (∀ (index : Index.t) (use : Typing_env.t),
       Joined_envs.get_nth_joined_env envs index = Some use ->
       typing_env_kinded use) ->
    levels_found envs ->
    (∀ (index : Index.t) (use : Typing_env.t) (x : variable)
       (kind : flambda_kind) (ty : flambda_type)
       (binding_time_and_mode : Binding_time.With_name_mode.t),
       Joined_envs.get_nth_joined_env envs index = Some use ->
       simple_of_kind use (Var x) kind ->
       Keyed.lookup name_eqb (Name_var x) (Typing_env.names_to_types source)
       = Some (ty, binding_time_and_mode) ->
       FlambdaTypeGrammar.kind ty = kind) ->
    Join_env.join_aliases_into_bindings p fuel
      (Join_env.create envs
         (Bindings_in_target_env.from_source_env avoid source))
      envs
    = Some s1 ->
    Join_env.n_way_join_delayed_equations fuel choices jt s1 []
      (Join_env.new_equations_in_joined_envs s1
         (Join_env.create envs
            (Bindings_in_target_env.from_source_env avoid source)))
    = Some (relations2, s2) ->
    Join_env.n_way_join_loop (Join_env.join_depth p) p fuel choices jt s2
      relations2
      (Join_env.new_definitions_of_existentials s2
         (Join_env.create envs
            (Bindings_in_target_env.from_source_env avoid source)))
    = Some (relations3, s3) ->
    state_true source s3
    ∧ relations_hold s3 relations3
    ∧ state_grows Join_env.joined_env Join_env.definitions_of_existentials
        kind_in_target_env
        (Join_env.create envs
           (Bindings_in_target_env.from_source_env avoid source))
        s3.

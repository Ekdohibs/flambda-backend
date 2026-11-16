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



(* What the functions that the meet is made of keep, one function at a
   time.

   This file is to be reviewed: the statements, which are about the
   functions of FlambdaMeet.v. The contract of the whole meet is in
   FlambdaMeetSpec.v; it is to be proved from statements of this kind,
   and the join of types uses some of these functions too. The proofs
   are in FlambdaMeetStepsProofs.v, which needs no review.

   What is stated so far.
   - [reduce_head_of_kind_naked_immediate] loses no value
     ([reduce_head_of_kind_naked_immediate_sound_statement]): under a
     valuation that satisfies the environment strictly, a value that
     has a head of naked immediates has the head that the function
     answers, and the function does not answer bottom. Nothing else is
     asked of the environment or of the head. The function reads, for
     each relation of the head, the canonical element of the name that
     the relation is on: a constant gives the number, by the
     compiler's reading of a relation on a constant
     ([Relation.of_const]), and the relation goes; a name replaces the
     name. That the number so read is the number that the relation
     relates the constant to is a statement of
     FlambdaTypeMeaningSpec.v ([relation_of_const_sound_statement]).
   - What it is for the meet of a part of a type to be right
     ([meet_call_ok], [part_meet_ok]): the two halves of the contract
     of the whole meet, for a part of any type, given what it is for
     such a part to fit an environment and to be true of the value in
     hand. The meets of parts are put together from one another, and
     each statement below says that one way of putting them together
     keeps this: [map_result] ([map_result_ok_statement]),
     [meet_unknown] ([meet_unknown_ok_statement]), [combine_results2]
     ([combine_results2_ok_statement]). An answer that is one of the
     inputs is right because the inputs stay fitting and true as the
     environment and the valuation are extended ([part_stable]);
     nothing is said of how precise an answer is.
   - Sets: [set_meet] answers what both sets have
     ([set_meet_cases_statement]); the meet of two heads of naked
     numbers is right ([naked_number_meet_ok_statement]). *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaClosuresSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaEnvAliases
  FlambdaEnvAliasesSpec FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaMeetGrammar FlambdaExpandHead FlambdaMeetEnv FlambdaMeet
  FlambdaMeetSpec.
Import ListNotations.

(* ======================================================================== *)
(* The relations of a naked immediate, reduced                              *)
(* ======================================================================== *)

(* [reduce_head_of_kind_naked_immediate] loses no value. *)
Definition reduce_head_of_kind_naked_immediate_sound_statement : Prop :=
  ∀ (statics : static_env) (valuation : env) (tenv : Typing_env.t)
    (head : head_of_kind_naked_immediate) (v : value),
    typing_env_satisfied_strictly statics valuation tenv ->
    head_of_kind_naked_immediate_satisfied statics valuation v head ->
    ∃ head' : head_of_kind_naked_immediate,
      reduce_head_of_kind_naked_immediate tenv head = Or_bottom.Ok head'
      ∧ head_of_kind_naked_immediate_satisfied statics valuation v head'.

(* ======================================================================== *)
(* The meet of a part of a type                                             *)
(* ======================================================================== *)

(* The meet of two types is made of the meets of their parts: sets of
   numbers, products of types, the cases of a row-like structure, and
   so on. Each such meet is given the environment of the meet and
   answers one of its inputs or a new thing, with the environment that
   it has added to. What follows says what it is for one of them to
   keep what [meet_expanded_head_keeps] and [meet_expanded_head_sound]
   (FlambdaMeetSpec.v) ask of the whole, for a part of any type [A],
   given what it is for a part to fit an environment ([fits]: it is
   well formed and the environment knows its names, at their kinds)
   and to be true of the value in hand under a valuation
   ([satisfied]). *)

Section Part.

Context {A : Type}.

Variable fits : Typing_env.t -> A -> Prop.
Variable satisfied : static_env -> env -> A -> Prop.

(* What is asked of [fits] and [satisfied]: a part that fits an
   environment fits one that extends it, and is true under a valuation
   that agrees, on what the environment knows, with one under which it
   is true. *)
Definition part_stable : Prop :=
  (∀ (menv menv' : Meet_env.t) (a : A),
     meet_env_extends menv menv' ->
     fits (Meet_env.typing_env menv) a ->
     fits (Meet_env.typing_env menv') a)
  ∧ (∀ (statics : static_env) (tenv : Typing_env.t)
       (valuation valuation' : env) (a : A),
       fits tenv a ->
       extended_from tenv valuation valuation' ->
       satisfied statics valuation a ->
       satisfied statics valuation' a).

(* What one call of a meet answered is right. If it is not bottom, the
   environment answered is the first extended, it fits, the part
   answered fits it, and the variables that it knows and the first
   did not are new ([new_names_avoided]). And no value is lost: with
   a static environment that is valid ([statics_valid],
   FlambdaClosuresSpec.v; FlambdaMeetSpec.v says why), under a
   valuation that satisfies the environment and under which both
   inputs are true, the answer is not bottom, and a valuation that
   agrees with the first on what the environment knew satisfies the
   environment answered and makes the part answered true; and no
   equation has been added on a name that has no value
   ([keeps_equations_valued]). *)
Definition meet_call_ok (menv : Meet_env.t) (a1 a2 : A)
    (result : Meet_result.t A) : Prop :=
  let tenv := Meet_env.typing_env menv in
  (∀ (r : meet_return_value A) (menv' : Meet_env.t),
     result = Meet_result.Ok r menv' ->
     meet_env_extends menv menv'
     ∧ typing_env_fits (Meet_env.typing_env menv')
     ∧ fits (Meet_env.typing_env menv') (extract_value r a1 a2)
     ∧ new_names_avoided menv menv')
  ∧ ∀ (statics : static_env) (valuation : env),
      statics_valid statics ->
      typing_env_satisfied_strictly statics valuation tenv ->
      satisfied statics valuation a1 ->
      satisfied statics valuation a2 ->
      ∃ (r : meet_return_value A) (menv' : Meet_env.t) (valuation' : env),
        result = Meet_result.Ok r menv'
        ∧ extended_from tenv valuation valuation'
        ∧ typing_env_satisfied_strictly statics valuation'
            (Meet_env.typing_env menv')
        ∧ satisfied statics valuation' (extract_value r a1 a2)
        ∧ keeps_equations_valued tenv (Meet_env.typing_env menv') valuation
            valuation'.

(* A meet of parts is right: every call of it is, on inputs that fit
   an environment that fits. Nothing is said of a call that does not
   answer ([None]: a fatal error of the compiler, not enough fuel, or
   the state at which the model stops and the compiler does not,
   FlambdaMeetEnv.v). *)
Definition part_meet_ok
    (meet : Meet_env.t -> A -> A -> option (Meet_result.t A)) : Prop :=
  ∀ (menv : Meet_env.t) (a1 a2 : A) (result : Meet_result.t A),
    typing_env_fits (Meet_env.typing_env menv) ->
    fits (Meet_env.typing_env menv) a1 ->
    fits (Meet_env.typing_env menv) a2 ->
    meet menv a1 a2 = Some result ->
    meet_call_ok menv a1 a2 result.

End Part.

(* A thing that may be unknown: it fits, and is true, if it is unknown
   or what is known does and is. *)
Definition or_unknown_fits {A : Type} (fits : Typing_env.t -> A -> Prop)
    (tenv : Typing_env.t) (x : Or_unknown.t A) : Prop :=
  match x with
  | Or_unknown.Unknown => True
  | Or_unknown.Known a => fits tenv a
  end.

Definition or_unknown_satisfied {A : Type}
    (satisfied : static_env -> env -> A -> Prop) (statics : static_env)
    (valuation : env) (x : Or_unknown.t A) : Prop :=
  match x with
  | Or_unknown.Unknown => True
  | Or_unknown.Known a => satisfied statics valuation a
  end.

(* [map_result]: the answer of a meet of parts, as the answer of the
   meet of what the parts are made into by [f]. *)
Definition map_result_ok_statement : Prop :=
  ∀ (A B : Type)
    (fits_a : Typing_env.t -> A -> Prop)
    (satisfied_a : static_env -> env -> A -> Prop)
    (fits_b : Typing_env.t -> B -> Prop)
    (satisfied_b : static_env -> env -> B -> Prop) (f : A -> B)
    (menv : Meet_env.t) (a1 a2 : A) (result : Meet_result.t A),
    (∀ (tenv : Typing_env.t) (a : A), fits_a tenv a -> fits_b tenv (f a)) ->
    (∀ (statics : static_env) (valuation : env) (a : A),
       satisfied_b statics valuation (f a) <-> satisfied_a statics valuation a)
    ->
    meet_call_ok fits_a satisfied_a menv a1 a2 result ->
    meet_call_ok fits_b satisfied_b menv (f a1) (f a2)
      (map_result f result).

(* [meet_unknown]. What it is given to say that a thing known is
   bottom must be right: such a thing is true of nothing. *)
Definition meet_unknown_ok_statement : Prop :=
  ∀ (A : Type)
    (fits : Typing_env.t -> A -> Prop)
    (satisfied : static_env -> env -> A -> Prop)
    (meet_contents : Meet_env.t -> A -> A -> option (Meet_result.t A))
    (contents_is_bottom : A -> bool),
    (∀ (statics : static_env) (valuation : env) (a : A),
       contents_is_bottom a = true -> ~ satisfied statics valuation a) ->
    part_meet_ok fits satisfied meet_contents ->
    part_meet_ok (or_unknown_fits fits)
      (or_unknown_satisfied satisfied)
      (meet_unknown meet_contents contents_is_bottom).

(* [combine_results2]: the meet of two things that are each made of two
   parts, by the meets of the parts, one after the other. The thing
   made of two parts that fit must fit, and it must be true exactly
   when its parts are. *)
Definition combine_results2_ok_statement : Prop :=
  ∀ (A B C : Type)
    (fits_a : Typing_env.t -> A -> Prop)
    (satisfied_a : static_env -> env -> A -> Prop)
    (fits_b : Typing_env.t -> B -> Prop)
    (satisfied_b : static_env -> env -> B -> Prop)
    (fits_c : Typing_env.t -> C -> Prop)
    (satisfied_c : static_env -> env -> C -> Prop)
    (meet_a : Meet_env.t -> A -> A -> option (Meet_result.t A))
    (meet_b : Meet_env.t -> B -> B -> option (Meet_result.t B))
    (rebuild : A -> B -> C) (menv : Meet_env.t) (left_a right_a : A)
    (left_b right_b : B) (result : Meet_result.t C),
    let tenv := Meet_env.typing_env menv in
    part_stable fits_a satisfied_a ->
    part_stable fits_b satisfied_b ->
    part_meet_ok fits_a satisfied_a meet_a ->
    part_meet_ok fits_b satisfied_b meet_b ->
    (∀ (tenv : Typing_env.t) (a : A) (b : B),
       fits_a tenv a -> fits_b tenv b -> fits_c tenv (rebuild a b)) ->
    (∀ (statics : static_env) (valuation : env) (a : A) (b : B),
       satisfied_c statics valuation (rebuild a b)
       <-> satisfied_a statics valuation a ∧ satisfied_b statics valuation b)
    ->
    typing_env_fits tenv ->
    fits_a tenv left_a ->
    fits_a tenv right_a ->
    fits_b tenv left_b ->
    fits_b tenv right_b ->
    combine_results2 menv meet_a meet_b left_a right_a left_b right_b rebuild
    = Some result ->
    meet_call_ok fits_c satisfied_c menv
      (rebuild left_a left_b) (rebuild right_a right_b) result.

(* [meet_or_unknown_or_bottom]: the meet of two things of which either
   may be unknown or bottom, by a meet of the two where both are known.
   The answer is of a type [B] that such a thing is made into by [inj]
   (itself, or an expanded type); what is made of bottom is true of
   nothing. *)
Definition meet_or_unknown_or_bottom_ok_statement : Prop :=
  ∀ (A B : Type)
    (fits : Typing_env.t -> B -> Prop)
    (satisfied : static_env -> env -> B -> Prop)
    (inj : or_unknown_or_bottom A -> B)
    (meet_elt : Meet_env.t -> A -> A -> option (Meet_result.t B))
    (menv : Meet_env.t) (input1 input2 : or_unknown_or_bottom A)
    (result : Meet_result.t B),
    let tenv := Meet_env.typing_env menv in
    (∀ (a1 a2 : A) (result : Meet_result.t B),
       fits tenv (inj (Ok a1)) ->
       fits tenv (inj (Ok a2)) ->
       meet_elt menv a1 a2 = Some result ->
       meet_call_ok fits satisfied menv (inj (Ok a1))
         (inj (Ok a2)) result) ->
    (∀ (statics : static_env) (valuation : env),
       ~ satisfied statics valuation (inj Bottom)) ->
    typing_env_fits tenv ->
    fits tenv (inj input1) ->
    fits tenv (inj input2) ->
    meet_or_unknown_or_bottom meet_elt menv input1 input2 = Some result ->
    meet_call_ok fits satisfied menv (inj input1)
      (inj input2) result.

(* ======================================================================== *)
(* Products                                                                 *)
(* ======================================================================== *)

(* [meet_mapping]: the meet of two products, component by component.
   The components of the two are read by [get_left] and [get_right] at
   the keys that the fold of the caller goes through, in order. What
   it is for a component to be true may depend on its key
   ([satisfied]). The statement says what the call answers where it
   answers a new product: [rebuild] of a list that has one component
   for each key, in the reverse of that order, each fitting the
   environment answered and, in the second half, true under the
   valuation answered. Of an answer that is an input nothing is said
   here: an input stays as it was. *)
Definition meet_mapping_ok_statement : Prop :=
  ∀ (K A M : Type)
    (fits : Typing_env.t -> A -> Prop)
    (satisfied : K -> static_env -> env -> A -> Prop)
    (meet_data : Meet_env.t -> A -> A -> option (Meet_result.t A))
    (fold2 :
      (K -> option A -> option A -> mapping_state K A -> mapping_state K A) ->
      M -> M -> mapping_state K A -> mapping_state K A)
    (keys : list K) (get_left get_right : K -> option A) (menv : Meet_env.t)
    (left right : M) (rebuild : list (K * A) -> M)
    (result : Meet_result.t M),
    let tenv := Meet_env.typing_env menv in
    (∀ (f :
          K -> option A -> option A -> mapping_state K A ->
          mapping_state K A)
       (init : mapping_state K A),
       fold2 f left right init
       = List.fold_left
           (fun state key => f key (get_left key) (get_right key) state)
           keys init) ->
    (∀ key : K, part_stable fits (satisfied key)) ->
    (∀ key : K,
       part_meet_ok fits (satisfied key) meet_data) ->
    typing_env_fits tenv ->
    (∀ (key : K) (a : A), get_left key = Some a -> fits tenv a) ->
    (∀ (key : K) (a : A), get_right key = Some a -> fits tenv a) ->
    meet_mapping meet_data fold2 menv left right rebuild = Some result ->
    (∀ (r : meet_return_value M) (menv' : Meet_env.t),
       result = Meet_result.Ok r menv' ->
       meet_env_extends menv menv'
       ∧ typing_env_fits (Meet_env.typing_env menv')
       ∧ new_names_avoided menv menv'
       ∧ ∀ m : M,
           r = New_result m ->
           ∃ l : list (K * A),
             m = rebuild l
             ∧ map fst l = rev keys
             ∧ ∀ (key : K) (a : A),
                 (key, a) ∈ l -> fits (Meet_env.typing_env menv') a)
    ∧ ∀ (statics : static_env) (valuation : env),
        statics_valid statics ->
        typing_env_satisfied_strictly statics valuation tenv ->
        (∀ (key : K) (a : A),
           get_left key = Some a -> satisfied key statics valuation a) ->
        (∀ (key : K) (a : A),
           get_right key = Some a -> satisfied key statics valuation a) ->
        ∃ (r : meet_return_value M) (menv' : Meet_env.t) (valuation' : env),
          result = Meet_result.Ok r menv'
          ∧ extended_from tenv valuation valuation'
          ∧ typing_env_satisfied_strictly statics valuation'
              (Meet_env.typing_env menv')
          ∧ keeps_equations_valued tenv (Meet_env.typing_env menv') valuation
              valuation'
          ∧ ∀ m : M,
              r = New_result m ->
              ∃ l : list (K * A),
                m = rebuild l
                ∧ map fst l = rev keys
                ∧ ∀ (key : K) (a : A),
                    (key, a) ∈ l ->
                    fits (Meet_env.typing_env menv') a
                    ∧ satisfied key statics valuation' a.

(* ======================================================================== *)
(* Sets                                                                     *)
(* ======================================================================== *)

(* [set_meet] answers bottom if the two sets have nothing in common,
   and otherwise a set that has exactly what both have; the
   environment is the one given. The set answered is not empty if
   neither input is. (An input that is empty is answered as it is, and
   not as bottom: a head that is a set is to be not empty.) *)
Definition set_meet_cases_statement : Prop :=
  ∀ (K B : Type) (eqb : K -> K -> bool) (of_set : list K -> B)
    (env : Meet_env.t) (s1 s2 : list K),
    decides_equality eqb ->
    match set_meet eqb of_set env s1 s2 with
    | Meet_result.Bottom _ => ∀ key : K, ~ (key ∈ s1 ∧ key ∈ s2)
    | Meet_result.Ok r env' =>
      env' = env
      ∧ ∃ s : list K,
          extract_value r (of_set s1) (of_set s2) = of_set s
          ∧ (∀ key : K, key ∈ s <-> key ∈ s1 ∧ key ∈ s2)
          ∧ (s1 <> [] -> s2 <> [] -> s <> [])
    end.

(* The meet of two heads of naked numbers of a kind other than the
   immediates: the sets of numbers. A head fits if it is not empty. *)
Definition naked_number_meet_ok_statement : Prop :=
  ∀ (k : naked_number_kind) (v : value),
    part_meet_ok
      (fun (_ : Typing_env.t) (numbers : list Z) => numbers <> [])
      (fun (_ : static_env) (_ : env) (numbers : list Z) =>
         naked_numbers_satisfied k v numbers)
      (fun (env : Meet_env.t) (t1 t2 : list Z) =>
         Some (meet_head_of_kind_naked_number env t1 t2)).

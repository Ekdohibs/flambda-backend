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


(* Proofs of the statements of FlambdaMeetHeadsSpec.v. Nothing here
   needs review. *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaPrimitiveProofs.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs
  FlambdaTypeMeaningSpec FlambdaTypeMeaningProofs FlambdaTypeMeaningBoolSpec
  FlambdaTypeMeaningBoolProofs FlambdaTypeEqualSpec FlambdaTypeEqualProofs
  FlambdaTypeWellFormedSpec FlambdaTypeWellFormedProofs FlambdaBindingTime
  FlambdaEnvAliases FlambdaEnvAliasesSpec FlambdaTypingEnv
  FlambdaTypingEnvSpec FlambdaTypingEnvInvariantSpec
  FlambdaTypingEnvKindedSpec FlambdaMeetGrammar
  FlambdaExpandHead FlambdaMeetEnv FlambdaMeet FlambdaMeetSpec
  FlambdaMeetStepsSpec FlambdaMeetStepsProofs FlambdaMeetEnvSpec
  FlambdaMeetEnvProofs FlambdaMeetScopesSpec FlambdaMeetScopesProofs
  FlambdaMeetProductsSpec FlambdaMeetProductsProofs FlambdaMeetHeadsSpec.
Import ListNotations.

(* The proofs below, as far as the end of the section, are for any
   predicate on static environments that implies validity
   (FlambdaMeetValidity.v): in them the names of the four contracts
   that suppose a valid static environment stand for the same with
   that predicate. The theorems of the signed statements are after the
   section, at validity itself. *)
From OxCamlFlambda2Types Require Import FlambdaMeetValidity.

Section Validity.

Context {V : validity}.

Abbreviation statics_valid := valid.
Abbreviation meet_expanded_head_sound := meet_expanded_head_sound_for.
Abbreviation join_scoped_envs_sound := join_scoped_envs_sound_for.
Abbreviation meet_call_ok := meet_call_ok_for.
Abbreviation part_meet_ok := part_meet_ok_for.
Abbreviation meet_disjunction_right := meet_disjunction_right_for.
Abbreviation type_part_meet_ok := type_part_meet_ok_for.
Abbreviation meet_row_like_for_closures_right :=
  meet_row_like_for_closures_right_for.
Abbreviation value_heads_right := value_heads_right_for.

(* ======================================================================== *)
(* Meets of parts, put together                                             *)
(* ======================================================================== *)

(* A part of a type stays fitting and true as a type does. *)
Lemma type_part_stable {X : Type} (to_type : X -> flambda_type) (v : value) :
  part_stable (fits_as_type to_type) (satisfied_as_type to_type v).
Proof.
  split.
  - intros menv menv' x Hextends Hfits.
    exact (type_fits_extends menv menv' (to_type x) Hextends Hfits).
  - intros statics tenv valuation valuation' x Hfits Hextended Hsatisfied.
    exact (type_satisfied_extended statics tenv valuation valuation' v
             (to_type x) Hfits Hextended Hsatisfied).
Qed.

(* A call that is right for two inputs, a way of fitting and a way of
   being true is right for others, if they cannot be told apart. *)
Lemma meet_call_ok_change {A : Type} (fits fits' : Typing_env.t -> A -> Prop)
    (satisfied satisfied' : static_env -> env -> A -> Prop)
    (menv : Meet_env.t) (a1 a2 a1' a2' : A) (result : Meet_result.t A) :
  meet_call_ok fits satisfied menv a1 a2 result ->
  (∀ (r : meet_return_value A) (menv' : Meet_env.t),
     result = Meet_result.Ok r menv' ->
     fits (Meet_env.typing_env menv') (extract_value r a1 a2) ->
     fits' (Meet_env.typing_env menv') (extract_value r a1' a2')) ->
  (∀ (statics : static_env) (valuation : env),
     satisfied' statics valuation a1' -> satisfied statics valuation a1) ->
  (∀ (statics : static_env) (valuation : env),
     satisfied' statics valuation a2' -> satisfied statics valuation a2) ->
  (∀ (r : meet_return_value A) (statics : static_env) (valuation : env),
     satisfied statics valuation (extract_value r a1 a2) ->
     satisfied' statics valuation (extract_value r a1' a2')) ->
  meet_call_ok fits' satisfied' menv a1' a2' result.
Proof.
  intros [Hkeeps Hsound] Hfits H1 H2 Hanswer. split.
  - intros r menv' Hresult.
    destruct (Hkeeps r menv' Hresult) as (Hextends & Henv' & Hfitting & Hnames).
    split; [ exact Hextends | ]. split; [ exact Henv' | ].
    split; [ exact (Hfits r menv' Hresult Hfitting) | exact Hnames ].
  - intros statics valuation Hstatics Hvaluation Hs1 Hs2.
    destruct (Hsound statics valuation Hstatics Hvaluation (H1 _ _ Hs1)
                (H2 _ _ Hs2))
      as (r & menv' & valuation' & Hresult & Hextended & Hvaluation' & Hs
          & Hvalued).
    exists r, menv', valuation'. split; [ exact Hresult | ].
    split; [ exact Hextended | ]. split; [ exact Hvaluation' | ].
    split; [ exact (Hanswer r statics valuation' Hs) | exact Hvalued ].
Qed.

(* A part is true for the one thing that the static environment
   chooses, if it is true for each thing. *)
Definition chosen_satisfied {P X : Type} (choose : static_env -> P -> Prop)
    (satisfied : P -> static_env -> env -> X -> Prop) (statics : static_env)
    (valuation : env) (x : X) : Prop :=
  ∃ p : P, choose statics p ∧ satisfied p statics valuation x.

Lemma part_meet_ok_chosen {P X : Type} (fits : Typing_env.t -> X -> Prop)
    (choose : static_env -> P -> Prop)
    (satisfied : P -> static_env -> env -> X -> Prop)
    (meet : Meet_env.t -> X -> X -> option (Meet_result.t X)) (p0 : P) :
  (∀ (statics : static_env) (p p' : P),
     choose statics p -> choose statics p' -> p = p') ->
  (∀ p : P, part_meet_ok fits (satisfied p) meet) ->
  part_meet_ok fits (chosen_satisfied choose satisfied) meet.
Proof.
  intros Hfunctional Hok menv a1 a2 result Henv H1 H2 Hmeet. split.
  - exact (proj1 (Hok p0 menv a1 a2 result Henv H1 H2 Hmeet)).
  - intros statics valuation Hstatics Hvaluation (p & Hp & Hs1)
      (p' & Hp' & Hs2).
    assert (Hsame : p' = p) by exact (Hfunctional statics p' p Hp' Hp).
    subst p'.
    destruct (proj2 (Hok p menv a1 a2 result Henv H1 H2 Hmeet) statics
                valuation Hstatics Hvaluation Hs1 Hs2)
      as (r & menv' & valuation' & Hresult & Hextended & Hvaluation' & Hanswer
          & Hvalued).
    exists r, menv', valuation'. split; [ exact Hresult | ].
    split; [ exact Hextended | ]. split; [ exact Hvaluation' | ].
    split; [ | exact Hvalued ]. exists p. split; [ exact Hp | exact Hanswer ].
Qed.

(* The meet of two types, for the one value that the static environment
   chooses. *)
Lemma meet_type_chosen_ok (mh : meet_expanded_head)
    (choose : static_env -> value -> Prop) :
  meet_expanded_head_keeps mh ->
  meet_expanded_head_sound mh ->
  (∀ (statics : static_env) (w w' : value),
     choose statics w -> choose statics w' -> w = w') ->
  part_meet_ok type_fits
    (chosen_satisfied choose
       (fun (w : value) (statics : static_env) (valuation : env)
            (ty : flambda_type) =>
          type_satisfied statics valuation w ty))
    (meet mh).
Proof.
  intros Hkeeps Hsound Hfunctional.
  apply (part_meet_ok_chosen type_fits choose _ (meet mh) (Value_const Null)
           Hfunctional).
  intros w. exact (meet_type_ok_for mh w Hkeeps Hsound).
Qed.

(* ======================================================================== *)
(* The variables that stand for the result of a primitive                   *)
(* ======================================================================== *)

Lemma relation_variable_part_stable (number : static_env -> option Z) :
  part_stable relation_variable_fits (relation_variable_satisfied number).
Proof.
  split.
  - intros menv menv' [ x | ] Hextends; [ | intros _; exact I ].
    intros [Hmem Hkind]. split.
    + exact (tenv_extends_mem _ _ _ (meet_env_extends_tenv _ _ Hextends) Hmem).
    + exact (proj1 (proj2 (proj2 (proj2 (proj2 Hextends)))) (Var x) _ Hkind).
  - intros statics tenv valuation valuation' [ x | ] Hfits Hextended
      (n & Hn & Hx); exists n; (split; [ exact Hn | ]); [ | exact I ].
    cbn in *. rewrite (Hextended x (proj1 Hfits)). exact Hx.
Qed.

Lemma meet_relation_ok_for :
  for_validity meet_relation_ok_statement.
Proof.
  intros mh number Hkeeps Hsound menv x1 x2 result Henv H1 H2 Hmeet.
  unfold meet_relation in Hmeet.
  destruct x1 as [ x1 | ], x2 as [ x2 | ].
  - destruct (Meet_env.add_alias menv (Var x1) (Var x2) mh)
      as [ added | ] eqn:Hadd; [ | discriminate Hmeet ].
    cbn in Hmeet. destruct H1 as [Hmem1 Hkind1], H2 as [Hmem2 Hkind2].
    split.
    + intros r menv' ->.
      destruct added as [ menv1 | ]; [ | discriminate Hmeet ].
      injection Hmeet as <- <-.
      destruct (add_alias_keeps_for mh menv menv1 (Var x1) (Var x2) _ Hkeeps
                  Henv
                  Hkind1 Hkind2 Hadd)
        as [[Hextends Henv'] Hnames].
      split; [ exact Hextends | ]. split; [ exact Henv' | ].
      split; [ | exact Hnames ].
      exact (proj1 (relation_variable_part_stable number) menv menv1
               (Some x1) Hextends (conj Hmem1 Hkind1)).
    + intros statics valuation Hstatics Hvaluation (n & Hn & Hs1)
        (n' & Hn' & Hs2).
      rewrite Hn in Hn'. injection Hn' as <-. cbn in Hs1, Hs2.
      destruct (meet_env_add_alias_sound_for mh statics valuation menv (Var x1)
                  (Var x2) _ (Value_const (Naked_immediate n)) added Hkeeps
                  Hsound Henv Hkind1 Hkind2 Hstatics Hvaluation Hs1 Hs2 Hadd)
        as (menv' & -> & valuation' & Hextended & Hvaluation' & Hvalued).
      injection Hmeet as <-.
      exists Both_inputs, menv', valuation'. split; [ reflexivity | ].
      split; [ exact Hextended | ]. split; [ exact Hvaluation' | ].
      split; [ | exact Hvalued ].
      exists n. split; [ exact Hn | ]. cbn. rewrite (Hextended x1 Hmem1).
      exact Hs1.
  - injection Hmeet as <-.
    apply (meet_call_ok_same_env _ _ menv _ _ Left_input Henv H1).
    intros statics valuation _ Hs1 _. exact Hs1.
  - injection Hmeet as <-.
    apply (meet_call_ok_same_env _ _ menv _ _ Right_input Henv H2).
    intros statics valuation _ _ Hs2. exact Hs2.
  - injection Hmeet as <-.
    apply (meet_call_ok_same_env _ _ menv _ _ Both_inputs Henv H1).
    intros statics valuation _ Hs1 _. exact Hs1.
Qed.

(* ======================================================================== *)
(* What a disjunction answers for each of its two parts                     *)
(* ======================================================================== *)

(* The local functions of [meet_disjunction], as functions of their
   own. *)
Definition disjunction_when_a (extensions : variant_extensions)
    : option env_extension :=
  match extensions with
  | No_extensions => None
  | Ext when_a _ => Some when_a
  end.

Definition disjunction_when_b (extensions : variant_extensions)
    : option env_extension :=
  match extensions with
  | No_extensions => None
  | Ext _ when_b => Some when_b
  end.

Definition disjunction_add_extension (mh : meet_expanded_head)
    (env : Or_bottom.t Meet_env.t) (extension : option env_extension)
    : option (Or_bottom.t Meet_env.t) :=
  match extension, env with
  | None, _ => Some env
  | Some _, Or_bottom.Bottom => Some Or_bottom.Bottom
  | Some extension, Or_bottom.Ok env =>
    Meet_env.add_env_extension_strict env extension mh
  end.

Definition disjunction_meet_side {C : Type}
    (meet_c : Meet_env.t -> C -> C -> option (Meet_result.t C))
    (env : Or_bottom.t Meet_env.t) (val1 val2 : C)
    : option (Meet_result.t C) :=
  match env with
  | Or_bottom.Bottom => Some (Meet_result.Bottom (New_result tt))
  | Or_bottom.Ok env =>
    let* result := meet_c env val1 val2 in
    match result with
    | Meet_result.Bottom r => Some (Meet_result.Bottom r)
    | Meet_result.Ok result env =>
      match Meet_env.final_typing_env_strict env with
      | Or_bottom.Bottom => Some (Meet_result.Bottom (New_result tt))
      | Or_bottom.Ok _ => Some (Meet_result.Ok result env)
      end
    end
  end.

Definition disjunction_direct_return {A B : Type} (choices : meet_choices)
    (mh : meet_expanded_head) (initial_env : Meet_env.t) (join_scope : Scope.t)
    (result : meet_return_value (A * B * variant_extensions))
    (scoped_env : Meet_env.t)
    : option (Meet_result.t (A * B * variant_extensions)) :=
  let* level := Typing_env.cut (Meet_env.typing_env scoped_env) join_scope in
  let* initial_env :=
    add_defined_vars choices
      (Meet_env.with_names_to_avoid initial_env
         (Meet_env.names_to_avoid scoped_env))
      level
  in
  let ext := extension_of_level level in
  let* added := Meet_env.add_env_extension_strict initial_env ext mh in
  match added with
  | Or_bottom.Bottom =>
    Some (Meet_result.Bottom (map_return_value (fun _ => tt) result))
  | Or_bottom.Ok env => Some (Meet_result.Ok result env)
  end.

(* What [meet_disjunction] does with the answers of the two sides. *)
Definition disjunction_finish {A B : Type} (join : join_scoped_envs)
    (choices : meet_choices) (mh : meet_expanded_head)
    (bottom_a : unit -> A) (bottom_b : unit -> B) (initial_env : Meet_env.t)
    (join_scope : Scope.t) (initial_tenv : Typing_env.t) (val_a1 : A)
    (val_b1 : B) (extensions1 : variant_extensions) (val_a2 : A) (val_b2 : B)
    (extensions2 : variant_extensions) (a_result : Meet_result.t A)
    (b_result : Meet_result.t B)
    : option (Meet_result.t (A * B * variant_extensions)) :=
  match a_result, b_result with
  | Meet_result.Bottom r1, Meet_result.Bottom r2 =>
    Some
      (Meet_result.Bottom (combine_meet_return_values r1 r2 (fun _ => tt)))
  | Meet_result.Ok a_result env, Meet_result.Bottom b =>
    let result :=
      combine_meet_return_values a_result b
        (fun _ =>
           let val_a := extract_value a_result val_a1 val_a2 in
           let val_b := bottom_b tt in
           (val_a, val_b, No_extensions))
    in
    disjunction_direct_return choices mh initial_env join_scope result env
  | Meet_result.Bottom a, Meet_result.Ok b_result env =>
    let result :=
      combine_meet_return_values a b_result
        (fun _ =>
           let val_b := extract_value b_result val_b1 val_b2 in
           let val_a := bottom_a tt in
           (val_a, val_b, No_extensions))
    in
    disjunction_direct_return choices mh initial_env join_scope result env
  | Meet_result.Ok a_result env_a, Meet_result.Ok b_result env_b =>
    let tenv_a := Meet_env.final_typing_env env_a in
    let tenv_b := Meet_env.final_typing_env env_b in
    let* result_env :=
      join mh join_scope
        (Meet_env.with_names_to_avoid initial_env
           (Meet_env.names_to_avoid env_b))
        initial_tenv [tenv_a; tenv_b]
    in
    let* when_a_level := Typing_env.cut tenv_a join_scope in
    let* when_b_level := Typing_env.cut tenv_b join_scope in
    let* result_env := add_defined_vars choices result_env when_a_level in
    let* result_env := add_defined_vars choices result_env when_b_level in
    let extensions :=
      if Typing_env_level.is_empty when_a_level
         && Typing_env_level.is_empty when_b_level
      then No_extensions
      else
        let when_a := extension_of_level when_a_level in
        let when_b := extension_of_level when_b_level in
        Ext when_a when_b
    in
    let env_extension_result : meet_return_value unit :=
      match extensions, extensions1, extensions2 with
      | No_extensions, No_extensions, No_extensions => Both_inputs
      | No_extensions, No_extensions, Ext _ _ => Left_input
      | No_extensions, Ext _ _, No_extensions => Right_input
      | (No_extensions | Ext _ _), _, _ => New_result tt
      end
    in
    let result :=
      combine_meet_return_values
        (combine_meet_return_values a_result b_result (fun _ => tt))
        env_extension_result
        (fun _ =>
           let val_a := extract_value a_result val_a1 val_a2 in
           let val_b := extract_value b_result val_b1 val_b2 in
           (val_a, val_b, extensions))
    in
    Some (Meet_result.Ok result result_env)
  end.

Lemma meet_disjunction_unfold {A B : Type} (join : join_scoped_envs)
    (choices : meet_choices) (mh : meet_expanded_head)
    (meet_a : Meet_env.t -> A -> A -> option (Meet_result.t A))
    (meet_b : Meet_env.t -> B -> B -> option (Meet_result.t B))
    (bottom_a : unit -> A) (bottom_b : unit -> B) (initial_env : Meet_env.t)
    (val_a1 : A) (val_b1 : B) (extensions1 : variant_extensions) (val_a2 : A)
    (val_b2 : B) (extensions2 : variant_extensions) :
  meet_disjunction join choices mh meet_a meet_b bottom_a bottom_b
    initial_env val_a1 val_b1 extensions1 val_a2 val_b2 extensions2
  = let '(join_scope, initial_tenv, env) :=
      Meet_env.enter_scope initial_env
    in
    let* env_a :=
      disjunction_add_extension mh (Or_bottom.Ok env)
        (disjunction_when_a extensions1)
    in
    let* env_a :=
      disjunction_add_extension mh env_a (disjunction_when_a extensions2)
    in
    let* a_result := disjunction_meet_side meet_a env_a val_a1 val_a2 in
    let env :=
      match a_result with
      | Meet_result.Bottom _ => env
      | Meet_result.Ok _ env_a =>
        Meet_env.with_names_to_avoid env (Meet_env.names_to_avoid env_a)
      end
    in
    let* env_b :=
      disjunction_add_extension mh (Or_bottom.Ok env)
        (disjunction_when_b extensions1)
    in
    let* env_b :=
      disjunction_add_extension mh env_b (disjunction_when_b extensions2)
    in
    let* b_result := disjunction_meet_side meet_b env_b val_b1 val_b2 in
    disjunction_finish join choices mh bottom_a bottom_b initial_env
      join_scope initial_tenv val_a1 val_b1 extensions1 val_a2 val_b2
      extensions2 a_result b_result.
Proof. reflexivity. Qed.

(* An answer of two meets put together is an input, or the new thing. *)
Lemma combine_answer_cases {A B C : Type} (ra : meet_return_value A)
    (rb : meet_return_value B) (f : unit -> C) (left right answer : C) :
  meet_answer (combine_meet_return_values ra rb f) left right answer ->
  answer = left ∨ answer = right ∨ answer = f tt.
Proof. destruct ra, rb; cbn; intuition. Qed.

Lemma disjunction_meet_side_ok {C : Type}
    (meet_c : Meet_env.t -> C -> C -> option (Meet_result.t C))
    (env : Or_bottom.t Meet_env.t) (val1 val2 : C) (r : meet_return_value C)
    (env' : Meet_env.t) :
  disjunction_meet_side meet_c env val1 val2 = Some (Meet_result.Ok r env') ->
  ∃ env0 : Meet_env.t, meet_c env0 val1 val2 = Some (Meet_result.Ok r env').
Proof.
  unfold disjunction_meet_side. destruct env as [ env0 | ]; [ | discriminate ].
  destruct (meet_c env0 val1 val2) as [ [ r0 | r0 env1 ] | ] eqn:Hmeet;
    try discriminate.
  cbn. intros [= <- <-]. exists env0. exact Hmeet.
Qed.

Lemma disjunction_direct_return_ok {A B : Type} (choices : meet_choices)
    (mh : meet_expanded_head) (initial_env : Meet_env.t) (join_scope : Scope.t)
    (result r : meet_return_value (A * B * variant_extensions))
    (scoped_env menv' : Meet_env.t) :
  disjunction_direct_return choices mh initial_env join_scope result
    scoped_env
  = Some (Meet_result.Ok r menv') ->
  r = result.
Proof.
  unfold disjunction_direct_return.
  destruct (Typing_env.cut (Meet_env.typing_env scoped_env) join_scope)
    as [ level | ]; [ | discriminate ].
  destruct (add_defined_vars choices _ level) as [ env1 | ];
    [ | discriminate ].
  cbv zeta.
  destruct (Meet_env.add_env_extension_strict env1 _ mh) as [ [ env2 | ] | ];
    try discriminate.
  intros [= <- <-]. reflexivity.
Qed.

Lemma meet_disjunction_answer_for :
  for_validity meet_disjunction_answer_statement.
Proof.
  intros A B join choices mh meet_a meet_b bottom_a bottom_b menv menv' a1 a2
    a b1 b2 b extensions1 extensions2 extensions r Hmeet Hanswer.
  rewrite meet_disjunction_unfold in Hmeet.
  destruct (Meet_env.enter_scope menv) as [[join_scope initial_tenv] env].
  destruct (disjunction_add_extension mh (Or_bottom.Ok env)
              (disjunction_when_a extensions1))
    as [ env_a0 | ]; [ | discriminate Hmeet ].
  destruct (disjunction_add_extension mh env_a0
              (disjunction_when_a extensions2))
    as [ env_a | ]; [ | discriminate Hmeet ].
  destruct (disjunction_meet_side meet_a env_a a1 a2) as [ a_result | ]
    eqn:Ha; [ | discriminate Hmeet ].
  cbv zeta in Hmeet.
  match type of Hmeet with
  | context [disjunction_add_extension mh ?e
               (disjunction_when_b extensions1)] =>
    destruct (disjunction_add_extension mh e (disjunction_when_b extensions1))
      as [ env_b0 | ]; [ | discriminate Hmeet ]
  end.
  destruct (disjunction_add_extension mh env_b0
              (disjunction_when_b extensions2))
    as [ env_b | ]; [ | discriminate Hmeet ].
  destruct (disjunction_meet_side meet_b env_b b1 b2) as [ b_result | ]
    eqn:Hb; [ | discriminate Hmeet ].
  unfold disjunction_finish in Hmeet.
  assert (Hinput :
    (a, b, extensions) = (a1, b1, extensions1)
    ∨ (a, b, extensions) = (a2, b2, extensions2) ->
    side_answer meet_a bottom_a a1 a2 a ∧ side_answer meet_b bottom_b b1 b2 b).
  { unfold side_answer. intros [ [= -> -> ->] | [= -> -> ->] ]; auto. }
  destruct a_result as [ ra | ra env_a' ], b_result as [ rb | rb env_b' ].
  - discriminate Hmeet.
  - cbv zeta in Hmeet. apply disjunction_direct_return_ok in Hmeet.
    subst r. apply combine_answer_cases in Hanswer.
    destruct Hanswer as [ H | [ H | H ] ]; [ auto | auto | ].
    cbv beta in H. injection H as -> -> ->.
    destruct (disjunction_meet_side_ok _ _ _ _ _ _ Hb) as [env0 Hmeet_b].
    split.
    + right. right. left. reflexivity.
    + right. right. right. exists env0, env_b', rb. auto.
  - cbv zeta in Hmeet. apply disjunction_direct_return_ok in Hmeet.
    subst r. apply combine_answer_cases in Hanswer.
    destruct Hanswer as [ H | [ H | H ] ]; [ auto | auto | ].
    cbv beta in H. injection H as -> -> ->.
    destruct (disjunction_meet_side_ok _ _ _ _ _ _ Ha) as [env0 Hmeet_a].
    split.
    + right. right. right. exists env0, env_a', ra. auto.
    + right. right. left. reflexivity.
  - cbv zeta in Hmeet.
    destruct (join mh join_scope _ initial_tenv _) as [ result_env | ];
      [ | discriminate Hmeet ].
    destruct (Typing_env.cut (Meet_env.final_typing_env env_a') join_scope)
      as [ when_a_level | ]; [ | discriminate Hmeet ].
    destruct (Typing_env.cut (Meet_env.final_typing_env env_b') join_scope)
      as [ when_b_level | ]; [ | discriminate Hmeet ].
    destruct (add_defined_vars choices result_env when_a_level)
      as [ result_env1 | ]; [ | discriminate Hmeet ].
    destruct (add_defined_vars choices result_env1 when_b_level)
      as [ result_env2 | ]; [ | discriminate Hmeet ].
    injection Hmeet as <- <-. apply combine_answer_cases in Hanswer.
    destruct Hanswer as [ H | [ H | H ] ]; [ auto | auto | ].
    cbv beta in H. injection H as -> -> ->.
    destruct (disjunction_meet_side_ok _ _ _ _ _ _ Ha) as [env0 Hmeet_a].
    destruct (disjunction_meet_side_ok _ _ _ _ _ _ Hb) as [env1 Hmeet_b].
    split.
    + right. right. right. exists env0, env_a', ra. auto.
    + right. right. right. exists env1, env_b', rb. auto.
Qed.

(* ======================================================================== *)
(* The heads that are not null                                              *)
(* ======================================================================== *)

(* A value has the type of the values that are not null and have a
   head. *)
Lemma non_null_value_satisfied (statics : static_env) (vars : env) (v : value)
    (head : head_of_kind_value_non_null) :
  type_satisfied statics vars v (non_null_value head)
  <-> v <> Value_const Null
      ∧ head_of_kind_value_non_null_satisfied statics vars v head.
Proof.
  unfold non_null_value. rewrite type_satisfied_value.
  unfold descr_satisfied. rewrite head_of_kind_value_satisfied_eq. tauto.
Qed.

(* Two heads that a value has are made with compatible constructors. *)
Lemma satisfied_heads_compatible (statics : static_env) (vars : env)
    (v : value) (head head' : head_of_kind_value_non_null) :
  satisfied_as_type non_null_value v statics vars head ->
  satisfied_as_type non_null_value v statics vars head' ->
  heads_compatible head head'.
Proof.
  unfold satisfied_as_type. rewrite !non_null_value_satisfied.
  intros [_ H] [_ H'].
  exact (heads_compatible_satisfied statics vars v head head' H H').
Qed.

(* No value has the type of a mutable block. *)
Lemma mutable_block_not_satisfied (statics : static_env) (vars : env)
    (v : value) (alloc_mode : Alloc_mode.t) :
  satisfied_as_type non_null_value v statics vars (Mutable_block alloc_mode) ->
  False.
Proof.
  unfold satisfied_as_type. rewrite non_null_value_satisfied.
  rewrite mutable_block_head_satisfied. tauto.
Qed.

(* The type of a mutable block fits every environment. *)
Lemma mutable_block_fits (tenv : Typing_env.t) (alloc_mode : Alloc_mode.t) :
  fits_as_type non_null_value tenv (Mutable_block alloc_mode).
Proof.
  unfold fits_as_type, type_fits, type_kinded, type_equations_kinded. cbn.
  split; [ reflexivity | ].
  split; [ intros x [] | ]. split; [ intros s kind [] | intros n kind [] ].
Qed.

(* The type of a boxed number fits when the type of its contents does,
   and is of the kind boxed. *)
Lemma boxed_number_fits (tenv : Typing_env.t) (k : boxable_number)
    (contents : flambda_type) (alloc_mode : Alloc_mode.t) :
  type_fits tenv (non_null_value (Boxed_number k contents alloc_mode))
  <-> kind contents = unboxed_kind k ∧ type_fits tenv contents.
Proof.
  unfold type_fits, type_kinded, type_equations_kinded. cbn.
  rewrite !app_nil_r, andb_true_iff, kind_eqb_eq. tauto.
Qed.

(* The type of strings fits when there is a string. *)
Lemma string_fits (tenv : Typing_env.t) (strs : list (list Z)) :
  type_fits tenv (non_null_value (String strs)) <-> strs <> [].
Proof.
  unfold type_fits, type_kinded, type_equations_kinded. cbn.
  rewrite non_empty_iff. split; [ tauto | ]. intros Hstrs.
  split; [ exact Hstrs | ].
  split; [ intros x [] | ]. split; [ intros s kind [] | intros n kind [] ].
Qed.

(* The meet of two allocation modes: nothing is asked of a mode. *)
Lemma meet_alloc_mode_ok :
  part_meet_ok (fun (_ : Typing_env.t) (_ : Alloc_mode.t) => True)
    (fun (_ : static_env) (_ : env) (_ : Alloc_mode.t) => True)
    (fun (env : Meet_env.t) (alloc_mode1 alloc_mode2 : Alloc_mode.t) =>
       Some (meet_alloc_mode env alloc_mode1 alloc_mode2)).
Proof.
  intros menv a1 a2 result Henv _ _ [= <-].
  destruct a1, a2; cbn; apply meet_call_ok_same_env; auto.
Qed.

Lemma alloc_mode_part_stable :
  part_stable (fun (_ : Typing_env.t) (_ : Alloc_mode.t) => True)
    (fun (_ : static_env) (_ : env) (_ : Alloc_mode.t) => True).
Proof. split; intros; exact I. Qed.

Lemma lists_z_eqb_decides_equality : decides_equality (lists_eqb Z.eqb).
Proof.
  intros l l'. apply lists_eqb_eq. intros a a' _. apply Z.eqb_eq.
Qed.

Section Non_null.

Variable join : join_scoped_envs.
Variable choices : meet_choices.
Variable mh : meet_expanded_head.
Variable v : value.

Hypothesis Hkeeps : meet_expanded_head_keeps mh.
Hypothesis Hsound : meet_expanded_head_sound mh.

(* The contents of a boxed number. *)
Lemma meet_boxed_contents_ok (k : boxable_number) :
  type_part_meet_ok
    (fun contents : flambda_type =>
       non_null_value (Boxed_number k contents Alloc_mode.Heap))
    v (meet mh).
Proof.
  intros menv t1 t2 result Henv H1 H2 Hmeet.
  apply boxed_number_fits in H1 as [Hk1 Hf1].
  apply boxed_number_fits in H2 as [Hk2 Hf2].
  set (choose :=
    fun (statics : static_env) (w : value) =>
      v <> Value_const Null
      ∧ ∃ c : const,
          boxed_number_of statics v = Some c
          ∧ kind_of_const c = unboxed_kind k
          ∧ w = Value_const c).
  assert (Hfunctional :
    ∀ (statics : static_env) (w w' : value),
      choose statics w -> choose statics w' -> w = w').
  { intros statics w w' (_ & c & Hc & _ & ->) (_ & c' & Hc' & _ & ->).
    congruence. }
  assert (Hiff :
    ∀ (statics : static_env) (valuation : env) (ty : flambda_type),
      satisfied_as_type
        (fun contents : flambda_type =>
           non_null_value (Boxed_number k contents Alloc_mode.Heap))
        v statics valuation ty
      <-> chosen_satisfied choose
            (fun (w : value) (statics : static_env) (valuation : env)
                 (ty : flambda_type) =>
               type_satisfied statics valuation w ty)
            statics valuation ty).
  { intros statics valuation ty. unfold satisfied_as_type, chosen_satisfied.
    rewrite non_null_value_satisfied, boxed_number_satisfied. split.
    - intros (Hnot_null & c & Hc & Hkind & Hty). exists (Value_const c).
      split; [ | exact Hty ]. split; [ exact Hnot_null | ]. exists c. auto.
    - intros (w & (Hnot_null & c & Hc & Hkind & ->) & Hty).
      split; [ exact Hnot_null | ]. exists c. auto. }
  apply (meet_call_ok_change type_fits _ _ _ menv t1 t2 t1 t2 result
           (meet_type_chosen_ok mh choose Hkeeps Hsound Hfunctional menv t1 t2
              result Henv Hf1 Hf2 Hmeet)).
  - intros r menv' -> Hanswer. apply boxed_number_fits.
    split; [ | exact Hanswer ].
    rewrite (proj2 (proj2 (meet_type_keeps_for mh menv menv' t1 t2 r Hkeeps Henv
                             Hf1 Hf2 Hmeet))).
    exact Hk1.
  - intros statics valuation. apply Hiff.
  - intros statics valuation. apply Hiff.
  - intros r statics valuation. apply Hiff.
Qed.

(* Sets of strings. *)
Lemma meet_strings_ok :
  type_part_meet_ok (fun strs : list (list Z) => non_null_value (String strs))
    v
    (fun (env : Meet_env.t) (strs1 strs2 : list (list Z)) =>
       Some (set_meet (lists_eqb Z.eqb) (fun s => s) env strs1 strs2)).
Proof.
  intros menv strs1 strs2 result Henv H1 H2 [= <-].
  apply string_fits in H1. apply string_fits in H2.
  assert (Hcases :=
    set_meet_cases_for (list Z) (list (list Z)) (lists_eqb Z.eqb) (fun s => s)
      menv strs1 strs2 lists_z_eqb_decides_equality).
  assert (Hboth :
    ∀ (statics : static_env) (valuation : env),
      satisfied_as_type (fun strs => non_null_value (String strs)) v statics
        valuation strs1 ->
      satisfied_as_type (fun strs => non_null_value (String strs)) v statics
        valuation strs2 ->
      v <> Value_const Null
      ∧ ∃ bytes : list Z,
          string_of statics v = Some bytes ∧ bytes ∈ strs1 ∧ bytes ∈ strs2).
  { intros statics valuation Hs1 Hs2. unfold satisfied_as_type in Hs1, Hs2.
    rewrite non_null_value_satisfied, string_satisfied in Hs1, Hs2.
    destruct Hs1 as (Hnot_null & bytes & Hbytes & Hin1).
    destruct Hs2 as (_ & bytes' & Hbytes' & Hin2).
    rewrite Hbytes in Hbytes'. injection Hbytes' as <-.
    split; [ exact Hnot_null | ]. exists bytes. auto. }
  destruct (set_meet (lists_eqb Z.eqb) (fun s : list (list Z) => s) menv strs1
              strs2)
    as [ r | r env' ].
  - apply meet_call_ok_bottom. intros statics valuation _ _ Hs1 Hs2.
    destruct (Hboth statics valuation Hs1 Hs2)
      as (_ & bytes & _ & Hin1 & Hin2).
    exact (Hcases bytes (conj Hin1 Hin2)).
  - destruct Hcases as (-> & s & Hextract & Hmem & Hne).
    apply meet_call_ok_same_env; [ exact Henv | | ].
    + rewrite Hextract. apply string_fits. exact (Hne H1 H2).
    + intros statics valuation _ Hs1 Hs2. rewrite Hextract.
      destruct (Hboth statics valuation Hs1 Hs2)
        as (Hnot_null & bytes & Hbytes & Hin1 & Hin2).
      unfold satisfied_as_type.
      rewrite non_null_value_satisfied, string_satisfied.
      split; [ exact Hnot_null | ]. exists bytes.
      split; [ exact Hbytes | ]. apply Hmem. split; assumption.
Qed.

End Non_null.

Lemma meet_head_of_kind_value_non_null_ok_for :
  for_validity meet_head_of_kind_value_non_null_ok_statement.
Proof.
  intros join choices mh v Hkeeps Hsound Hvariant Hclosures menv head1 head2
    result Henv H1 H2 Hmeet.
  unfold meet_head_of_kind_value_non_null in Hmeet.
  destruct head1 as [ is_int1 imms1 get_tag1 blocks1 extensions1 is_unique1
                    | alloc_mode1 | k1 n1 alloc_mode1
                    | by_function_slot1 alloc_mode1 | strs1 ],
           head2 as [ is_int2 imms2 get_tag2 blocks2 extensions2 is_unique2
                    | alloc_mode2 | k2 n2 alloc_mode2
                    | by_function_slot2 alloc_mode2 | strs2 ].
  all: cbv beta iota zeta in Hmeet.
  all: try (injection Hmeet as <-; apply meet_call_ok_bottom;
            intros statics valuation _ _ Hs1 Hs2;
            exact (satisfied_heads_compatible statics valuation v _ _ Hs1 Hs2)).
  - (* Two variants. *)
    destruct (meet_variant join choices mh menv is_int1 get_tag1 blocks1 imms1
                is_int2 get_tag2 blocks2 imms2 extensions1 extensions2)
      as [ result0 | ] eqn:Hmeet0; [ | discriminate Hmeet ].
    injection Hmeet as <-.
    set (parts1 := (is_int1, get_tag1, blocks1, imms1, extensions1)).
    set (parts2 := (is_int2, get_tag2, blocks2, imms2, extensions2)).
    assert (Hok :=
      Hvariant menv parts1 parts2 result0 Henv H1 H2 Hmeet0).
    apply (map_result_ok_for _ _ _ _ (fits_as_type non_null_value)
             (satisfied_as_type non_null_value v)
             (variant_of_parts (is_unique1 || is_unique2)) menv parts1 parts2
             result0)
      in Hok.
    + apply (meet_call_ok_change _ _ _ _ menv _ _ _ _ _ Hok).
      * intros r menv' _. destruct r; exact (fun H => H).
      * intros statics valuation H. exact H.
      * intros statics valuation H. exact H.
      * intros r statics valuation. destruct r; exact (fun H => H).
    + intros tenv parts. destruct parts as [[[[? ?] ?] ?] ?].
      exact (fun H => H).
    + intros statics valuation parts. destruct parts as [[[[? ?] ?] ?] ?].
      reflexivity.
  - (* A variant and a mutable block: no value has the second. *)
    assert (Hok :
      ∀ r : meet_return_value head_of_kind_value_non_null,
        (r = Right_input
         ∨ ∃ alloc_mode : Alloc_mode.t,
             r = New_result (Mutable_block alloc_mode)) ->
        meet_call_ok (fits_as_type non_null_value)
          (satisfied_as_type non_null_value v) menv
          (Variant is_int1 imms1 get_tag1 blocks1 extensions1 is_unique1)
          (Mutable_block alloc_mode2) (Meet_result.Ok r menv)).
    { intros r Hr. apply meet_call_ok_same_env; [ exact Henv | | ].
      - destruct Hr as [ -> | [alloc_mode ->] ];
          [ exact H2 | apply mutable_block_fits ].
      - intros statics valuation _ _ Hs2.
        destruct (mutable_block_not_satisfied _ _ _ _ Hs2). }
    destruct blocks1 as [ [known other alloc_mode_left] | ].
    + destruct alloc_mode_left, alloc_mode2; cbn in Hmeet;
        injection Hmeet as <-; apply Hok; eauto.
    + injection Hmeet as <-. apply Hok. auto.
  - (* A mutable block and a variant: no value has the first. *)
    assert (Hok :
      ∀ r : meet_return_value head_of_kind_value_non_null,
        (r = Left_input
         ∨ ∃ alloc_mode : Alloc_mode.t,
             r = New_result (Mutable_block alloc_mode)) ->
        meet_call_ok (fits_as_type non_null_value)
          (satisfied_as_type non_null_value v) menv
          (Mutable_block alloc_mode1)
          (Variant is_int2 imms2 get_tag2 blocks2 extensions2 is_unique2)
          (Meet_result.Ok r menv)).
    { intros r Hr. apply meet_call_ok_same_env; [ exact Henv | | ].
      - destruct Hr as [ -> | [alloc_mode ->] ];
          [ exact H1 | apply mutable_block_fits ].
      - intros statics valuation _ Hs1 _.
        destruct (mutable_block_not_satisfied _ _ _ _ Hs1). }
    destruct blocks2 as [ [known other alloc_mode_right] | ].
    + destruct alloc_mode1, alloc_mode_right; cbn in Hmeet;
        injection Hmeet as <-; apply Hok; eauto.
    + injection Hmeet as <-. apply Hok. auto.
  - (* Two mutable blocks: no value has either. *)
    injection Hmeet as <-.
    destruct alloc_mode1, alloc_mode2; cbn;
      (apply meet_call_ok_same_env;
       [ exact Henv
       | first [ exact H1 | exact H2 ]
       | intros statics valuation _ Hs1 _;
         destruct (mutable_block_not_satisfied _ _ _ _ Hs1) ]).
  - (* Two boxed numbers. *)
    destruct (boxable_number_eqb k1 k2) eqn:Hk.
    + apply boxable_number_eqb_eq in Hk. subst k2.
      apply boxed_number_fits in H1 as H1'.
      apply boxed_number_fits in H2 as H2'.
      refine (combine_results2_ok_for _ _ _
                (fits_as_type
                   (fun contents : flambda_type =>
                      non_null_value
                        (Boxed_number k1 contents Alloc_mode.Heap)))
                (satisfied_as_type
                   (fun contents : flambda_type =>
                      non_null_value
                        (Boxed_number k1 contents Alloc_mode.Heap))
                   v)
                (fun _ _ => True) (fun _ _ _ => True)
                (fits_as_type non_null_value)
                (satisfied_as_type non_null_value v) (meet mh) _
                (Boxed_number k1) menv n1 n2 alloc_mode1 alloc_mode2 result
                (type_part_stable _ v) alloc_mode_part_stable
                (meet_boxed_contents_ok mh v Hkeeps Hsound k1)
                meet_alloc_mode_ok _ _ Henv _ _ I I Hmeet).
      * intros tenv contents alloc_mode Hfits _.
        apply boxed_number_fits. apply boxed_number_fits in Hfits.
        exact Hfits.
      * intros statics valuation contents alloc_mode.
        unfold satisfied_as_type. rewrite !non_null_value_satisfied.
        rewrite !boxed_number_satisfied. tauto.
      * apply boxed_number_fits. exact H1'.
      * apply boxed_number_fits. exact H2'.
    + injection Hmeet as <-. apply meet_call_ok_bottom.
      intros statics valuation _ _ Hs1 Hs2.
      assert (Hsame :=
        satisfied_heads_compatible statics valuation v _ _ Hs1 Hs2).
      cbn in Hsame. apply boxable_number_eqb_eq in Hsame. congruence.
  - (* Two sets of closures. *)
    refine (combine_results2_ok_for _ _ _
              (fits_as_type (create_closures Alloc_mode.Heap))
              (satisfied_as_type (create_closures Alloc_mode.Heap) v)
              (fun _ _ => True) (fun _ _ _ => True)
              (fits_as_type non_null_value)
              (satisfied_as_type non_null_value v)
              (meet_row_like_for_closures join choices mh) _ Closures menv
              by_function_slot1 by_function_slot2 alloc_mode1 alloc_mode2
              result (type_part_stable _ v) alloc_mode_part_stable Hclosures
              meet_alloc_mode_ok _ _ Henv H1 H2 I I Hmeet).
    + intros tenv by_function_slot alloc_mode Hfits _. exact Hfits.
    + intros statics valuation by_function_slot alloc_mode.
      unfold satisfied_as_type, create_closures.
      rewrite !non_null_value_satisfied, !closures_satisfied. tauto.
  - (* Two sets of strings. *)
    injection Hmeet as <-.
    apply (map_result_ok_for _ _
             (fits_as_type
                (fun strs : list (list Z) => non_null_value (String strs)))
             (satisfied_as_type
                (fun strs : list (list Z) => non_null_value (String strs)) v)
             (fits_as_type non_null_value)
             (satisfied_as_type non_null_value v) String menv strs1 strs2).
    + intros tenv strs Hfits. exact Hfits.
    + intros statics valuation strs. reflexivity.
    + exact (meet_strings_ok v menv strs1 strs2 _ Henv H1 H2 eq_refl).
Qed.

(* ======================================================================== *)
(* What it is for a variant to fit, by its parts                            *)
(* ======================================================================== *)

(* What it is to fit, by the lists of a thing: it is well formed, the
   environment knows its variables, and its aliases and the names of
   its equations are at their kinds. [type_fits] is that of the lists
   of a type. *)
Definition lists_fit (tenv : Typing_env.t) (well_formed : bool)
    (variables : list variable) (aliases : list (simple * flambda_kind))
    (equations : list (name * flambda_kind)) : Prop :=
  well_formed = true
  ∧ (∀ x : variable,
      x ∈ variables -> Typing_env.mem None tenv (Name_var x) = true)
  ∧ (∀ (s : simple) (kind : flambda_kind),
      (s, kind) ∈ aliases -> simple_of_kind tenv s kind)
  ∧ ∀ (n : name) (kind : flambda_kind),
      (n, kind) ∈ equations -> simple_of_kind tenv (simple_of_name n) kind.

Lemma type_fits_lists (tenv : Typing_env.t) (ty : flambda_type) :
  type_fits tenv ty
  <-> lists_fit tenv (type_well_formed ty) (free_variables ty)
        (aliases_with_kinds ty) (equation_names_with_kinds ty).
Proof. reflexivity. Qed.

Lemma lists_fit_app (tenv : Typing_env.t) (w1 w2 : bool)
    (f1 f2 : list variable) (a1 a2 : list (simple * flambda_kind))
    (e1 e2 : list (name * flambda_kind)) :
  lists_fit tenv (w1 && w2) (f1 ++ f2) (a1 ++ a2) (e1 ++ e2)
  <-> lists_fit tenv w1 f1 a1 e1 ∧ lists_fit tenv w2 f2 a2 e2.
Proof.
  unfold lists_fit. rewrite andb_true_iff. split.
  - intros ([Hw1 Hw2] & Hf & Ha & He).
    split; (split; [ assumption | ]); (split; [ | split ]).
    + intros x Hx. apply Hf, in_or_app. left. exact Hx.
    + intros s k Hs. apply Ha, in_or_app. left. exact Hs.
    + intros n k Hn. apply He, in_or_app. left. exact Hn.
    + intros x Hx. apply Hf, in_or_app. right. exact Hx.
    + intros s k Hs. apply Ha, in_or_app. right. exact Hs.
    + intros n k Hn. apply He, in_or_app. right. exact Hn.
  - intros [(Hw1 & Hf1 & Ha1 & He1) (Hw2 & Hf2 & Ha2 & He2)].
    split; [ split; assumption | ]. split; [ | split ].
    + intros x Hx. apply in_app_or in Hx as [ Hx | Hx ]; auto.
    + intros s k Hs. apply in_app_or in Hs as [ Hs | Hs ]; auto.
    + intros n k Hn. apply in_app_or in Hn as [ Hn | Hn ]; auto.
Qed.

Lemma lists_fit_nil (tenv : Typing_env.t) : lists_fit tenv true [] [] [].
Proof.
  split; [ reflexivity | ]. split; [ intros x [] | ].
  split; [ intros s k [] | intros n k [] ].
Qed.

(* An extension inside a type fits when its lists do. *)
Lemma extension_lists_fit (tenv : Typing_env.t) (extension : env_extension) :
  extension_fits tenv extension ->
  lists_fit tenv (env_extension_well_formed extension)
    (free_variables_of_env_extension extension)
    (aliases_of_env_extension extension)
    (equation_names_of_env_extension extension).
Proof.
  destruct extension as [equations]. intros (Hwf & Hfits & Hkinded).
  assert (Hlookup :
    ∀ (n : name) (ty : flambda_type),
      (n, ty) ∈ equations -> Keyed.lookup name_eqb n equations = Some ty).
  { cbn in Hwf. apply andb_true_iff in Hwf as [Hdistinct _].
    exact (in_lookup_distinct name_eqb name_eqb_eq equations Hdistinct). }
  split; [ exact Hwf | ]. split; [ | split ].
  - intros x Hx. cbn in Hx. apply in_flat_map in Hx as ([n ty] & Hin & Hx).
    cbn in Hx. apply in_app_or in Hx.
    destruct (Hfits n ty (Hlookup n ty Hin)) as [Hmem Hty].
    destruct Hx as [ Hx | Hx ].
    + destruct n as [ y | s ]; cbn in Hx; [ | destruct Hx ].
      destruct Hx as [ <- | [] ]. exact Hmem.
    + exact (proj1 (proj2 Hty) x Hx).
  - intros s kind Hs. cbn in Hs. apply in_flat_map in Hs as ([n ty] & Hin & Hs).
    cbn in Hs.
    exact (proj1 (proj2 (proj2 (proj2 (Hfits n ty (Hlookup n ty Hin))))) s
             kind Hs).
  - intros n kind Hn. cbn in Hn.
    apply in_flat_map in Hn as ([n' ty] & Hin & Hn). cbn in Hn.
    destruct Hn as [ [= <- <-] | Hn ].
    + exact (Hkinded n' ty (Hlookup n' ty Hin)).
    + exact (proj2 (proj2 (proj2 (proj2 (Hfits n' ty (Hlookup n' ty Hin))))) n
               kind Hn).
Qed.

Lemma lists_fit_extension (tenv : Typing_env.t) (extension : env_extension) :
  typing_env_invariant tenv ->
  lists_fit tenv (env_extension_well_formed extension)
    (free_variables_of_env_extension extension)
    (aliases_of_env_extension extension)
    (equation_names_of_env_extension extension) ->
  extension_fits tenv extension.
Proof.
  destruct extension as [equations].
  intros Hinvariant (Hwf & Hfv & Hal & Heq).
  split; [ exact Hwf | ].
  assert (Hin :
    ∀ (n : name) (ty : flambda_type),
      Keyed.lookup name_eqb n equations = Some ty -> (n, ty) ∈ equations).
  { intros n ty. apply keyed_lookup_in. exact name_eqb_eq. }
  assert (Hwf_ty :
    ∀ (n : name) (ty : flambda_type),
      Keyed.lookup name_eqb n equations = Some ty ->
      type_well_formed ty = true).
  { cbn in Hwf. apply andb_true_iff in Hwf as [_ Hall].
    apply (keyed_for_all_bindings_b name name_eqb name_eqb_eq) in Hall.
    exact (proj1 (keyed_for_all_bindings_lookup name name_eqb name_eqb_eq _ _
                    equations)
             Hall). }
  split.
  - intros n ty Hlookup. assert (Hmember := Hin n ty Hlookup).
    assert (Hkind : simple_of_kind tenv (simple_of_name n) (kind ty)).
    { apply Heq. cbn. apply in_flat_map. exists (n, ty).
      split; [ exact Hmember | left; reflexivity ]. }
    split; [ exact (simple_of_kind_mem tenv n (kind ty) Hinvariant Hkind) | ].
    split; [ exact (Hwf_ty n ty Hlookup) | ]. split; [ | split ].
    + intros x Hx. apply Hfv. cbn. apply in_flat_map. exists (n, ty).
      split; [ exact Hmember | ]. cbn. apply in_or_app. right. exact Hx.
    + intros s k Hs. apply Hal. cbn. apply in_flat_map. exists (n, ty).
      split; [ exact Hmember | exact Hs ].
    + intros n' k Hn'. apply Heq. cbn. apply in_flat_map. exists (n, ty).
      split; [ exact Hmember | right; exact Hn' ].
  - intros n ty Hlookup. apply Heq. cbn. apply in_flat_map. exists (n, ty).
    split; [ exact (Hin n ty Hlookup) | left; reflexivity ].
Qed.

(* The parts of a variant. *)
Lemma relation_variable_lists_fit (tenv : Typing_env.t) (x : option variable) :
  relation_variable_fits tenv x
  <-> lists_fit tenv true (free_variables_of_option x) (aliases_of_relation x)
        [].
Proof.
  destruct x as [ x | ];
    unfold relation_variable_fits, lists_fit, free_variables_of_option,
      aliases_of_relation;
    split.
  - intros [Hmem Hkind]. split; [ reflexivity | ].
    split; [ intros y [ <- | [] ]; exact Hmem | ].
    split; [ intros s k [ [= <- <-] | [] ]; exact Hkind | intros n k [] ].
  - intros (_ & Hfv & Hal & _). split; [ apply Hfv; left; reflexivity | ].
    apply (Hal (Var x) (Naked_number Kind_naked_immediate)). left.
    reflexivity.
  - intros _. apply lists_fit_nil.
  - intros _. exact I.
Qed.

Lemma tag_immediate_lists_fit (tenv : Typing_env.t) (ty : flambda_type) :
  type_fits tenv (tag_immediate ty)
  <-> lists_fit tenv
        (kind_eqb (kind ty) (Naked_number Kind_naked_immediate)
         && type_well_formed ty)
        (free_variables ty) (aliases_with_kinds ty)
        (equation_names_with_kinds ty).
Proof.
  rewrite type_fits_lists. cbn. rewrite !app_nil_r, !andb_true_r.
  reflexivity.
Qed.

Lemma blocks_type_lists_fit (tenv : Typing_env.t)
    (blocks : row_like_for_blocks) :
  type_fits tenv (blocks_type blocks)
  <-> lists_fit tenv (row_like_for_blocks_well_formed blocks)
        (free_variables_of_row_like_for_blocks blocks)
        (aliases_of_row_like_for_blocks blocks)
        (equation_names_of_row_like_for_blocks blocks).
Proof.
  rewrite type_fits_lists. cbn. rewrite !app_nil_r, !andb_true_r.
  reflexivity.
Qed.

Lemma lists_fit_app_relation (tenv : Typing_env.t) (x : option variable)
    (w : bool) (f : list variable) (a : list (simple * flambda_kind))
    (e : list (name * flambda_kind)) :
  lists_fit tenv w (free_variables_of_option x ++ f)
    (aliases_of_relation x ++ a) e
  <-> relation_variable_fits tenv x ∧ lists_fit tenv w f a e.
Proof.
  rewrite relation_variable_lists_fit.
  exact (lists_fit_app tenv true w (free_variables_of_option x) f
           (aliases_of_relation x) a [] e).
Qed.

Lemma immediates_lists_fit (tenv : Typing_env.t)
    (imms : Or_unknown.t flambda_type) :
  lists_fit tenv
    match imms with
    | Or_unknown.Known ty =>
      kind_eqb (kind ty) (Naked_number Kind_naked_immediate)
      && type_well_formed ty
    | Or_unknown.Unknown => true
    end
    match imms with
    | Or_unknown.Known ty => free_variables ty
    | Or_unknown.Unknown => []
    end
    match imms with
    | Or_unknown.Known ty => aliases_with_kinds ty
    | Or_unknown.Unknown => []
    end
    match imms with
    | Or_unknown.Known ty => equation_names_with_kinds ty
    | Or_unknown.Unknown => []
    end
  <-> or_unknown_fits (fits_as_type tag_immediate) tenv imms.
Proof.
  destruct imms as [ ty | ].
  - symmetry. apply tag_immediate_lists_fit.
  - split; [ intros _; exact I | intros _; apply lists_fit_nil ].
Qed.

Lemma blocks_lists_fit (tenv : Typing_env.t)
    (blocks : Or_unknown.t row_like_for_blocks) :
  lists_fit tenv
    match blocks with
    | Or_unknown.Known blocks => row_like_for_blocks_well_formed blocks
    | Or_unknown.Unknown => true
    end
    match blocks with
    | Or_unknown.Known blocks => free_variables_of_row_like_for_blocks blocks
    | Or_unknown.Unknown => []
    end
    match blocks with
    | Or_unknown.Known blocks => aliases_of_row_like_for_blocks blocks
    | Or_unknown.Unknown => []
    end
    match blocks with
    | Or_unknown.Known blocks => equation_names_of_row_like_for_blocks blocks
    | Or_unknown.Unknown => []
    end
  <-> or_unknown_fits (fits_as_type blocks_type) tenv blocks.
Proof.
  destruct blocks as [ blocks | ].
  - symmetry. apply blocks_type_lists_fit.
  - split; [ intros _; exact I | intros _; apply lists_fit_nil ].
Qed.

(* The two extensions of a variant fit, by their lists. *)
Definition variant_extensions_lists_fit (tenv : Typing_env.t)
    (extensions : variant_extensions) : Prop :=
  match extensions with
  | No_extensions => True
  | Ext when_immediate when_block =>
    lists_fit tenv (env_extension_well_formed when_immediate)
      (free_variables_of_env_extension when_immediate)
      (aliases_of_env_extension when_immediate)
      (equation_names_of_env_extension when_immediate)
    ∧ lists_fit tenv (env_extension_well_formed when_block)
        (free_variables_of_env_extension when_block)
        (aliases_of_env_extension when_block)
        (equation_names_of_env_extension when_block)
  end.

Lemma extensions_lists_fit (tenv : Typing_env.t)
    (extensions : variant_extensions) :
  lists_fit tenv
    match extensions with
    | No_extensions => true
    | Ext when_immediate when_block =>
      env_extension_well_formed when_immediate
      && env_extension_well_formed when_block
    end
    match extensions with
    | No_extensions => []
    | Ext when_immediate when_block =>
      free_variables_of_env_extension when_immediate
      ++ free_variables_of_env_extension when_block
    end
    match extensions with
    | No_extensions => []
    | Ext when_immediate when_block =>
      aliases_of_env_extension when_immediate
      ++ aliases_of_env_extension when_block
    end
    match extensions with
    | No_extensions => []
    | Ext when_immediate when_block =>
      equation_names_of_env_extension when_immediate
      ++ equation_names_of_env_extension when_block
    end
  <-> variant_extensions_lists_fit tenv extensions.
Proof.
  destruct extensions as [ | when_immediate when_block ].
  - split; [ intros _; exact I | intros _; apply lists_fit_nil ].
  - apply lists_fit_app.
Qed.

(* A variant fits when its parts do. *)
Lemma variant_lists (tenv : Typing_env.t) (is_int : option variable)
    (imms : Or_unknown.t flambda_type) (get_tag : option variable)
    (blocks : Or_unknown.t row_like_for_blocks)
    (extensions : variant_extensions) (is_unique : bool) :
  type_fits tenv
    (non_null_value (Variant is_int imms get_tag blocks extensions is_unique))
  <-> relation_variable_fits tenv is_int
      ∧ or_unknown_fits (fits_as_type tag_immediate) tenv imms
      ∧ relation_variable_fits tenv get_tag
      ∧ or_unknown_fits (fits_as_type blocks_type) tenv blocks
      ∧ variant_extensions_lists_fit tenv extensions.
Proof.
  rewrite type_fits_lists. cbn. rewrite !app_nil_r, <- !andb_assoc.
  rewrite lists_fit_app_relation, lists_fit_app, lists_fit_app_relation,
    lists_fit_app.
  rewrite immediates_lists_fit, blocks_lists_fit, extensions_lists_fit.
  reflexivity.
Qed.

(* ======================================================================== *)
(* The two sides of a variant                                               *)
(* ======================================================================== *)

(* The side of the tagged immediates: the type of their numbers, if
   it is known, fits as the type of the tagged immediates whose
   numbers have it; it is true of the value in hand if the value is a
   tagged immediate and has that type. *)
Definition immediates_fits
    : Typing_env.t -> Or_unknown.t flambda_type -> Prop :=
  or_unknown_fits (fits_as_type tag_immediate).

Definition immediates_satisfied (v : value) (statics : static_env)
    (valuation : env) (imms : Or_unknown.t flambda_type) : Prop :=
  is_immediate v
  ∧ or_unknown_satisfied (satisfied_as_type tag_immediate v) statics valuation
      imms.

(* The side of the blocks: the variable for [Get_tag], and the blocks
   if they are known, as the type of the variants that are blocks among
   them. *)
Definition tagged_blocks_fits (tenv : Typing_env.t)
    (x : option variable * Or_unknown.t row_like_for_blocks) : Prop :=
  relation_variable_fits tenv (fst x)
  ∧ or_unknown_fits (fits_as_type blocks_type) tenv (snd x).

(* The tag of the value in hand, if it is a block. *)
Definition tag_number (v : value) (statics : static_env) : option Z :=
  option_map fst (block_of statics v).

(* The side of the blocks is true of the value in hand: the value is a
   block, the variable holds its tag, and the value has the type of the
   blocks. *)
Definition tagged_blocks_satisfied (v : value) (statics : static_env)
    (valuation : env)
    (x : option variable * Or_unknown.t row_like_for_blocks) : Prop :=
  relation_variable_satisfied (tag_number v) statics valuation (fst x)
  ∧ or_unknown_satisfied (satisfied_as_type blocks_type v) statics valuation
      (snd x).

(* What [Is_int] answers on the value in hand, if it is a block or a
   tagged immediate. *)
Definition is_int_number (v : value) (statics : static_env) : option Z :=
  match block_of statics v with
  | Some _ => Some 0%Z
  | None =>
    match v with
    | Value_const (Tagged_immediate _) => Some 1%Z
    | _ => None
    end
  end.

(* A variant that fits an environment has parts that fit, in the
   sense of the statement of the two alternatives: an extension inside
   a type fits in that sense, by lookups, when its lists do, the
   environment having its invariant (a symbol that is at a kind is
   known). And the converse. *)
Lemma variant_fits_parts (tenv : Typing_env.t) (is_int : option variable)
    (imms : Or_unknown.t flambda_type) (get_tag : option variable)
    (blocks : Or_unknown.t row_like_for_blocks)
    (extensions : variant_extensions) (is_unique : bool) :
  typing_env_invariant tenv ->
  type_fits tenv
    (non_null_value
       (Variant is_int imms get_tag blocks extensions is_unique)) ->
  relation_variable_fits tenv is_int
  ∧ disjunction_fits immediates_fits tagged_blocks_fits tenv
      (imms, (get_tag, blocks), extensions).
Proof.
  intros Hinvariant Hfits.
  apply variant_lists in Hfits
    as (Hint & Himms & Htag & Hblocks & Hextensions).
  split; [ exact Hint | ]. split; [ exact Himms | ].
  split; [ exact (conj Htag Hblocks) | ].
  destruct extensions as [ | when_immediate when_block ];
    [ split; exact I | ].
  destruct Hextensions as [Ha Hb].
  split; apply lists_fit_extension; assumption.
Qed.

Lemma parts_fit_variant (tenv : Typing_env.t) (is_int : option variable)
    (imms : Or_unknown.t flambda_type) (get_tag : option variable)
    (blocks : Or_unknown.t row_like_for_blocks)
    (extensions : variant_extensions) (is_unique : bool) :
  relation_variable_fits tenv is_int ->
  disjunction_fits immediates_fits tagged_blocks_fits tenv
    (imms, (get_tag, blocks), extensions) ->
  type_fits tenv
    (non_null_value (Variant is_int imms get_tag blocks extensions is_unique)).
Proof.
  intros Hint (Himms & [Htag Hblocks] & Ha & Hb).
  apply variant_lists.
  split; [ exact Hint | ]. split; [ exact Himms | ].
  split; [ exact Htag | ]. split; [ exact Hblocks | ].
  destruct extensions as [ | when_immediate when_block ]; [ exact I | ].
  split; apply extension_lists_fit; assumption.
Qed.

Lemma blocks_type_satisfied (statics : static_env) (vars : env) (v : value)
    (blocks : row_like_for_blocks) :
  type_satisfied statics vars v (blocks_type blocks)
  <-> ∃ (tag : Z) (fields : list value),
        block_of statics v = Some (tag, fields)
        ∧ row_like_for_blocks_satisfied statics vars tag fields blocks.
Proof.
  unfold blocks_type. rewrite (create_variant_satisfied statics vars v). split.
  - intros [ (i & _ & Hbottom & _) | (tag & fields & Hblock & Hrow & _) ].
    + destruct (bottom_satisfied _ _ _ _ Hbottom).
    + exists tag, fields. split; assumption.
  - intros (tag & fields & Hblock & Hrow). right. exists tag, fields.
    split; [ exact Hblock | ]. split; [ exact Hrow | exact I ].
Qed.

(* What it is for a value to have a variant, by its parts. *)
Lemma variant_satisfied_parts (statics : static_env) (valuation : env)
    (v : value) (is_int : option variable) (imms : Or_unknown.t flambda_type)
    (get_tag : option variable) (blocks : Or_unknown.t row_like_for_blocks)
    (extensions : variant_extensions) (is_unique : bool) :
  type_satisfied statics valuation v
    (non_null_value (Variant is_int imms get_tag blocks extensions is_unique))
  <-> relation_variable_satisfied (is_int_number v) statics valuation is_int
      ∧ disjunction_satisfied (immediates_satisfied v)
          (tagged_blocks_satisfied v) statics valuation
          (imms, (get_tag, blocks), extensions).
Proof.
  rewrite non_null_value_satisfied, variant_satisfied.
  unfold disjunction_satisfied, immediates_satisfied, tagged_blocks_satisfied,
    relation_variable_satisfied, satisfied_as_type.
  cbn [fst snd]. split.
  - intros [Hnot_null
              [ (i & -> & His_int & Himms & Hextension)
              | (tag & fields & Hblock & His_int & Hget_tag & Hblocks
                 & Hextension) ]].
    + split; [ exists 1%Z; split; [ reflexivity | exact His_int ] | ].
      left. split; [ split; [ exists i; reflexivity | ] | ].
      * destruct imms as [ ty | ]; [ | exact I ]. cbn.
        apply tag_immediate_satisfied. exists i.
        split; [ reflexivity | exact Himms ].
      * destruct extensions; exact Hextension.
    + split.
      { exists 0%Z. split; [ | exact His_int ].
        unfold is_int_number. rewrite Hblock. reflexivity. }
      right. split; [ split | ].
      * exists tag. split; [ | exact Hget_tag ].
        unfold tag_number. rewrite Hblock. reflexivity.
      * destruct blocks as [ blocks | ]; [ | exact I ]. cbn.
        apply blocks_type_satisfied. exists tag, fields.
        split; assumption.
      * destruct extensions; exact Hextension.
  - intros [(n & Hn & His_int)
              [ [[[i ->] Himms] Hextension]
              | [[(m & Hm & Hget_tag) Hblocks] Hextension] ]].
    + split; [ discriminate | ]. left. exists i.
      cbn in Hn. injection Hn as <-. split; [ reflexivity | ].
      split; [ exact His_int | ]. split.
      * destruct imms as [ ty | ]; [ | exact I ]. cbn in Himms.
        apply tag_immediate_satisfied in Himms as (i' & [= <-] & Himms).
        exact Himms.
      * destruct extensions; exact Hextension.
    + unfold tag_number in Hm.
      destruct (block_of statics v) as [ [tag fields] | ] eqn:Hblock;
        [ | discriminate Hm ].
      cbn in Hm. injection Hm as <-.
      split; [ intros ->; discriminate Hblock | ].
      right. exists tag, fields. split; [ reflexivity | ].
      unfold is_int_number in Hn. rewrite Hblock in Hn. injection Hn as <-.
      split; [ exact His_int | ]. split; [ exact Hget_tag | ]. split.
      * destruct blocks as [ blocks | ]; [ | exact I ]. cbn in Hblocks.
        apply blocks_type_satisfied in Hblocks
          as (tag' & fields' & Hblock' & Hrow).
        rewrite Hblock in Hblock'. injection Hblock' as <- <-. exact Hrow.
      * destruct extensions; exact Hextension.
Qed.

(* ======================================================================== *)
(* What the statement of the two alternatives asks of the two sides         *)
(* ======================================================================== *)

Lemma or_unknown_part_stable {A : Type} (fits : Typing_env.t -> A -> Prop)
    (satisfied : static_env -> env -> A -> Prop) :
  part_stable fits satisfied ->
  part_stable (or_unknown_fits fits) (or_unknown_satisfied satisfied).
Proof.
  intros [Hfits Hsatisfied]. split.
  - intros menv menv' [ a | ] Hextends H; [ | exact I ].
    exact (Hfits menv menv' a Hextends H).
  - intros statics tenv valuation valuation' [ a | ] H Hextended Hs;
      [ | exact I ].
    exact (Hsatisfied statics tenv valuation valuation' a H Hextended Hs).
Qed.

(* A meet that is right stays right where "true" also asks a fact of
   the static environment. *)
Lemma part_meet_ok_fact {A : Type} (fits : Typing_env.t -> A -> Prop)
    (satisfied : static_env -> env -> A -> Prop) (fact : static_env -> Prop)
    (meet : Meet_env.t -> A -> A -> option (Meet_result.t A)) :
  part_meet_ok fits satisfied meet ->
  part_meet_ok fits
    (fun (statics : static_env) (valuation : env) (a : A) =>
       fact statics ∧ satisfied statics valuation a)
    meet.
Proof.
  intros Hok menv a1 a2 result Henv H1 H2 Hmeet.
  destruct (Hok menv a1 a2 result Henv H1 H2 Hmeet) as [Hkeeps Hsound].
  split; [ exact Hkeeps | ].
  intros statics valuation Hstatics Hvaluation [Hfact Hs1] [_ Hs2].
  destruct (Hsound statics valuation Hstatics Hvaluation Hs1 Hs2)
    as (r & menv' & valuation' & Hresult & Hextended & Hvaluation' & Hanswer
        & Hvalued).
  exists r, menv', valuation'. split; [ exact Hresult | ].
  split; [ exact Hextended | ]. split; [ exact Hvaluation' | ].
  split; [ exact (conj Hfact Hanswer) | exact Hvalued ].
Qed.

Lemma immediates_part_stable (v : value) :
  part_stable immediates_fits (immediates_satisfied v).
Proof.
  destruct (or_unknown_part_stable _ _ (type_part_stable tag_immediate v))
    as [Hfits Hsatisfied].
  split; [ exact Hfits | ].
  intros statics tenv valuation valuation' imms H Hextended [Himmediate Hs].
  split; [ exact Himmediate | ].
  exact (Hsatisfied statics tenv valuation valuation' imms H Hextended Hs).
Qed.

Lemma tagged_blocks_part_stable (v : value) :
  part_stable tagged_blocks_fits (tagged_blocks_satisfied v).
Proof.
  destruct (relation_variable_part_stable (tag_number v)) as [Hfits1 Hs1].
  destruct (or_unknown_part_stable _ _ (type_part_stable blocks_type v))
    as [Hfits2 Hs2].
  split.
  - intros menv menv' [get_tag blocks] Hextends [Htag Hblocks].
    split; [ exact (Hfits1 menv menv' get_tag Hextends Htag) | ].
    exact (Hfits2 menv menv' blocks Hextends Hblocks).
  - intros statics tenv valuation valuation' [get_tag blocks] [Htag Hblocks]
      Hextended [Hs_tag Hs_blocks].
    split.
    + exact (Hs1 statics tenv valuation valuation' get_tag Htag Hextended
               Hs_tag).
    + exact (Hs2 statics tenv valuation valuation' blocks Hblocks Hextended
               Hs_blocks).
Qed.

Lemma relation_variable_fits_by_kinds : fits_by_kinds relation_variable_fits.
Proof.
  intros tenv tenv' [ x | ] [Hmem Hkind]; [ | intros _; exact I ].
  intros [Hx Hx_kind]. split; [ exact (Hmem _ Hx) | exact (Hkind _ _ Hx_kind) ].
Qed.

Lemma or_unknown_fits_by_kinds {A : Type} (fits : Typing_env.t -> A -> Prop) :
  fits_by_kinds fits -> fits_by_kinds (or_unknown_fits fits).
Proof.
  intros Hfits tenv tenv' [ a | ] Hknows H; [ | exact I ].
  exact (Hfits tenv tenv' a Hknows H).
Qed.

Lemma fits_as_type_by_kinds {X : Type} (to_type : X -> flambda_type) :
  fits_by_kinds (fits_as_type to_type).
Proof.
  intros tenv tenv' x Hknows H.
  exact (type_fits_by_kinds tenv tenv' (to_type x) Hknows H).
Qed.

Lemma immediates_fits_by_kinds : fits_by_kinds immediates_fits.
Proof. apply or_unknown_fits_by_kinds, fits_as_type_by_kinds. Qed.

Lemma tagged_blocks_fits_by_kinds : fits_by_kinds tagged_blocks_fits.
Proof.
  intros tenv tenv' [get_tag blocks] Hknows [Htag Hblocks]. split.
  - exact (relation_variable_fits_by_kinds tenv tenv' get_tag Hknows Htag).
  - exact (or_unknown_fits_by_kinds _ (fits_as_type_by_kinds blocks_type) tenv
             tenv' blocks Hknows Hblocks).
Qed.

Lemma sides_exclusive (v : value) (statics : static_env) (valuation : env)
    (imms : Or_unknown.t flambda_type)
    (x : option variable * Or_unknown.t row_like_for_blocks) :
  immediates_satisfied v statics valuation imms ->
  tagged_blocks_satisfied v statics valuation x ->
  False.
Proof. intros [[i ->] _] [(n & Hn & _) _]. discriminate Hn. Qed.

Lemma immediates_bottom_fits (tenv : Typing_env.t) :
  immediates_fits tenv
    (Or_unknown.Known (bottom (Naked_number Kind_naked_immediate))).
Proof. apply tag_immediate_lists_fit. apply lists_fit_nil. Qed.

Lemma tagged_blocks_bottom_fits (tenv : Typing_env.t) :
  tagged_blocks_fits tenv
    (None, Or_unknown.Known row_like_for_blocks_bottom).
Proof.
  split; [ exact I | ]. apply blocks_type_lists_fit. apply lists_fit_nil.
Qed.

(* ======================================================================== *)
(* The meets of the two sides, and of two variants                          *)
(* ======================================================================== *)

Lemma tag_immediate_fits (tenv : Typing_env.t) (ty : flambda_type) :
  type_fits tenv (tag_immediate ty)
  <-> kind ty = Naked_number Kind_naked_immediate ∧ type_fits tenv ty.
Proof.
  rewrite tag_immediate_lists_fit, type_fits_lists. unfold lists_fit.
  rewrite andb_true_iff, kind_eqb_eq. tauto.
Qed.

Section Variant.

Variable join : join_scoped_envs.
Variable choices : meet_choices.
Variable mh : meet_expanded_head.
Variable v : value.

Hypothesis Hjoin : join_scoped_envs_sound join.
Hypothesis Hdefine : define_again choices = false.
Hypothesis Hkeeps : meet_expanded_head_keeps mh.
Hypothesis Hsound : meet_expanded_head_sound mh.

(* The type of the numbers of the tagged immediates. *)
Lemma meet_tagged_immediate_ok : type_part_meet_ok tag_immediate v (meet mh).
Proof.
  intros menv t1 t2 result Henv H1 H2 Hmeet.
  apply tag_immediate_fits in H1 as [Hk1 Hf1].
  apply tag_immediate_fits in H2 as [Hk2 Hf2].
  set (choose :=
    fun (_ : static_env) (w : value) =>
      ∃ i : Z,
        v = Value_const (Tagged_immediate i)
        ∧ w = Value_const (Naked_immediate i)).
  assert (Hfunctional :
    ∀ (statics : static_env) (w w' : value),
      choose statics w -> choose statics w' -> w = w').
  { intros statics w w' (i & Hv & ->) (i' & Hv' & ->). congruence. }
  assert (Hiff :
    ∀ (statics : static_env) (valuation : env) (ty : flambda_type),
      satisfied_as_type tag_immediate v statics valuation ty
      <-> chosen_satisfied choose
            (fun (w : value) (statics : static_env) (valuation : env)
                 (ty : flambda_type) =>
               type_satisfied statics valuation w ty)
            statics valuation ty).
  { intros statics valuation ty. unfold satisfied_as_type, chosen_satisfied.
    rewrite (tag_immediate_satisfied statics valuation v). split.
    - intros (i & Hv & Hty). exists (Value_const (Naked_immediate i)).
      split; [ exists i; auto | exact Hty ].
    - intros (w & (i & Hv & ->) & Hty). exists i. auto. }
  apply (meet_call_ok_change type_fits _ _ _ menv t1 t2 t1 t2 result
           (meet_type_chosen_ok mh choose Hkeeps Hsound Hfunctional menv t1 t2
              result Henv Hf1 Hf2 Hmeet)).
  - intros r menv' -> Hanswer. apply tag_immediate_fits.
    split; [ | exact Hanswer ].
    rewrite (proj2 (proj2 (meet_type_keeps_for mh menv menv' t1 t2 r Hkeeps Henv
                             Hf1 Hf2 Hmeet))).
    exact Hk1.
  - intros statics valuation. apply Hiff.
  - intros statics valuation. apply Hiff.
  - intros r statics valuation. apply Hiff.
Qed.

Lemma meet_immediates_ok :
  part_meet_ok immediates_fits (immediates_satisfied v)
    (meet_unknown (meet mh) is_obviously_bottom).
Proof.
  apply (part_meet_ok_fact (or_unknown_fits (fits_as_type tag_immediate))
           (or_unknown_satisfied (satisfied_as_type tag_immediate v))
           (fun _ : static_env => is_immediate v)).
  apply meet_unknown_ok_for; [ | exact meet_tagged_immediate_ok ].
  intros statics valuation ty Hbottom Hs. unfold satisfied_as_type in Hs.
  apply tag_immediate_satisfied in Hs as (i & _ & Hs).
  exact (is_obviously_bottom_satisfied _ _ _ _ Hbottom Hs).
Qed.

Hypothesis Hblocks :
  type_part_meet_ok blocks_type v (meet_row_like_for_blocks join choices mh).

Lemma meet_tagged_blocks_ok :
  part_meet_ok tagged_blocks_fits (tagged_blocks_satisfied v)
    (fun (env : Meet_env.t)
         (left right : option variable * Or_unknown.t row_like_for_blocks) =>
       combine_results2 env (meet_relation mh)
         (meet_unknown (meet_row_like_for_blocks join choices mh)
            row_like_for_blocks_is_bottom)
         (fst left) (fst right) (snd left) (snd right)
         (fun get_tag blocks => (get_tag, blocks))).
Proof.
  intros menv [get_tag1 blocks1] [get_tag2 blocks2] result Henv [Hf1 Hb1]
    [Hf2 Hb2] Hmeet.
  cbn [fst snd] in *.
  refine (combine_results2_ok_for _ _ _ relation_variable_fits
            (relation_variable_satisfied (tag_number v))
            (or_unknown_fits (fits_as_type blocks_type))
            (or_unknown_satisfied (satisfied_as_type blocks_type v))
            tagged_blocks_fits (tagged_blocks_satisfied v) (meet_relation mh)
            _ (fun get_tag blocks => (get_tag, blocks)) menv get_tag1 get_tag2
            blocks1 blocks2 result (relation_variable_part_stable _)
            (or_unknown_part_stable _ _ (type_part_stable blocks_type v))
            (meet_relation_ok_for mh _ Hkeeps Hsound) _ _ _ Henv Hf1 Hf2 Hb1 Hb2
            Hmeet).
  - apply meet_unknown_ok_for; [ | exact Hblocks ].
    intros statics valuation blocks Hbottom Hs. unfold satisfied_as_type in Hs.
    apply blocks_type_satisfied in Hs as (tag & fields & _ & Hrow).
    exact (row_like_for_blocks_is_bottom_satisfied _ _ _ _ _ Hbottom Hrow).
  - intros tenv get_tag blocks Hf Hb. exact (conj Hf Hb).
  - intros statics valuation get_tag blocks. reflexivity.
Qed.

End Variant.

(* What is left of a variant when the variable for [Is_int] is set
   aside, in the order of [meet_variant]. *)
Definition variant_rest : Type :=
  option variable * Or_unknown.t row_like_for_blocks
  * Or_unknown.t flambda_type * variant_extensions.

Definition sides_of_rest (rest : variant_rest)
    : Or_unknown.t flambda_type
      * (option variable * Or_unknown.t row_like_for_blocks)
      * variant_extensions :=
  let '(get_tag, blocks, imms, extensions) := rest in
  (imms, (get_tag, blocks), extensions).

Lemma meet_variant_ok_for :
  for_validity meet_variant_ok_statement.
Proof.
  intros join choices mh v Hjoin Hdefine Hkeeps Hsound Hblocks menv
    [[[[is_int1 get_tag1] blocks1] imms1] extensions1]
    [[[[is_int2 get_tag2] blocks2] imms2] extensions2] result Henv H1 H2
    Hmeet.
  unfold fits_as_type in H1, H2. cbn [variant_of_parts] in H1, H2.
  apply (variant_fits_parts _ _ _ _ _ _ _ (proj1 Henv)) in H1
    as [Hint1 Hrest1].
  apply (variant_fits_parts _ _ _ _ _ _ _ (proj1 Henv)) in H2
    as [Hint2 Hrest2].
  cbn [meet_variant_parts] in Hmeet. unfold meet_variant in Hmeet.
  set (fits_rest :=
    fun (tenv : Typing_env.t) (rest : variant_rest) =>
      disjunction_fits immediates_fits tagged_blocks_fits tenv
        (sides_of_rest rest)).
  set (satisfied_rest :=
    fun (statics : static_env) (valuation : env) (rest : variant_rest) =>
      disjunction_satisfied (immediates_satisfied v)
        (tagged_blocks_satisfied v) statics valuation (sides_of_rest rest)).
  refine (combine_results2_ok_for _ _ _ relation_variable_fits
            (relation_variable_satisfied (is_int_number v)) fits_rest
            satisfied_rest
            (fits_as_type
               (fun parts : variant_parts =>
                  non_null_value (variant_of_parts false parts)))
            (satisfied_as_type
               (fun parts : variant_parts =>
                  non_null_value (variant_of_parts false parts))
               v)
            (meet_relation mh) _ _ menv is_int1 is_int2
            (get_tag1, blocks1, imms1, extensions1)
            (get_tag2, blocks2, imms2, extensions2) result
            (relation_variable_part_stable _) _
            (meet_relation_ok_for mh _ Hkeeps Hsound) _ _ _ Henv Hint1 Hint2
                Hrest1
            Hrest2 Hmeet).
  - (* The rest is stable. *)
    destruct (disjunction_stable immediates_fits (immediates_satisfied v)
                tagged_blocks_fits (tagged_blocks_satisfied v)
                (immediates_part_stable v) (tagged_blocks_part_stable v))
      as [Hfits Hsatisfied].
    split.
    + intros menv0 menv' rest. apply Hfits.
    + intros statics tenv valuation valuation' rest. apply Hsatisfied.
  - (* The meet of the rest is right. *)
    intros menv0 [[[gt1 b1] i1] e1] [[[gt2 b2] i2] e2] result0 Henv0 Hf1 Hf2
      Hmeet0.
    cbv beta iota in Hmeet0.
    match type of Hmeet0 with
    | option_map _ ?m = _ => destruct m as [ result1 | ] eqn:Hmeet1
    end; [ | discriminate Hmeet0 ].
    injection Hmeet0 as <-.
    refine (map_result_ok_for _ _
              (disjunction_fits immediates_fits tagged_blocks_fits)
              (disjunction_satisfied (immediates_satisfied v)
                 (tagged_blocks_satisfied v))
              fits_rest satisfied_rest
              (fun result : Or_unknown.t flambda_type
                            * (option variable
                               * Or_unknown.t row_like_for_blocks)
                            * variant_extensions =>
                 let '(imms, (get_tag, blocks), extensions) := result in
                 (get_tag, blocks, imms, extensions))
              menv0 (i1, (gt1, b1), e1) (i2, (gt2, b2), e2) result1 _ _ _).
    + intros tenv [[imms [get_tag blocks]] extensions] H. exact H.
    + intros statics valuation [[imms [get_tag blocks]] extensions].
      reflexivity.
    + exact (meet_disjunction_ok_for _ _ immediates_fits (immediates_satisfied
               v)
               tagged_blocks_fits (tagged_blocks_satisfied v) join choices mh
               _ _ _ _ Hjoin Hdefine Hkeeps Hsound (immediates_part_stable v)
               (tagged_blocks_part_stable v) immediates_fits_by_kinds
               tagged_blocks_fits_by_kinds
               (meet_immediates_ok mh v Hkeeps Hsound)
               (meet_tagged_blocks_ok join choices mh v Hkeeps Hsound Hblocks)
               (sides_exclusive v) immediates_bottom_fits
               tagged_blocks_bottom_fits menv0 (i1, (gt1, b1), e1)
               (i2, (gt2, b2), e2) result1 Henv0 Hf1 Hf2 Hmeet1).
  - (* A variant fits when its parts do. *)
    intros tenv is_int [[[get_tag blocks] imms] extensions] Hint Hrest.
    exact (parts_fit_variant tenv is_int imms get_tag blocks extensions false
             Hint Hrest).
  - (* A value has a variant when its parts are true. *)
    intros statics valuation is_int [[[get_tag blocks] imms] extensions].
    apply variant_satisfied_parts.
Qed.

(* ======================================================================== *)
(* The heads that may be null                                               *)
(* ======================================================================== *)

(* The two sides of a head of kind value, each as a type: the values
   that are not null and have what the head has for them; and null, if
   the head may be null, where its variable holds 1. *)
Definition not_null_type
    (non_null : or_unknown_or_bottom head_of_kind_value_non_null)
    : flambda_type :=
  Type_value (Ok (No_alias (Head_of_kind_value non_null Not_null))).

Definition null_type (is_null : is_null) : flambda_type :=
  Type_value (Ok (No_alias (Head_of_kind_value Bottom is_null))).

Lemma value_head_fits (tenv : Typing_env.t)
    (non_null : or_unknown_or_bottom head_of_kind_value_non_null)
    (is_null : is_null) :
  type_fits tenv (value_head_type (Head_of_kind_value non_null is_null))
  <-> type_fits tenv (not_null_type non_null)
      ∧ type_fits tenv (null_type is_null).
Proof.
  rewrite !type_fits_lists. cbn. rewrite !app_nil_r.
  rewrite <- lists_fit_app, andb_true_r, app_nil_r. reflexivity.
Qed.

Lemma null_type_fits (tenv : Typing_env.t) (x : option variable) :
  type_fits tenv (null_type (Maybe_null x)) <-> relation_variable_fits tenv x.
Proof.
  rewrite type_fits_lists. cbn. symmetry. apply relation_variable_lists_fit.
Qed.

Lemma not_null_type_satisfied (statics : static_env) (vars : env) (v : value)
    (non_null : or_unknown_or_bottom head_of_kind_value_non_null) :
  type_satisfied statics vars v (not_null_type non_null)
  <-> v <> Value_const Null
      ∧ match non_null with
        | Unknown => kind_of_value v = Value
        | Bottom => False
        | Ok head => head_of_kind_value_non_null_satisfied statics vars v head
        end.
Proof.
  unfold not_null_type. rewrite type_satisfied_value. unfold descr_satisfied.
  rewrite head_of_kind_value_satisfied_eq. tauto.
Qed.

Lemma null_type_satisfied (statics : static_env) (vars : env) (v : value)
    (is_null : is_null) :
  type_satisfied statics vars v (null_type is_null)
  <-> v = Value_const Null
      ∧ match is_null with
        | Not_null => False
        | Maybe_null x => variable_is vars x 1
        end.
Proof.
  unfold null_type. rewrite type_satisfied_value. unfold descr_satisfied.
  rewrite head_of_kind_value_satisfied_eq. tauto.
Qed.

Lemma value_head_satisfied (statics : static_env) (vars : env) (v : value)
    (non_null : or_unknown_or_bottom head_of_kind_value_non_null)
    (is_null : is_null) :
  type_satisfied statics vars v
    (value_head_type (Head_of_kind_value non_null is_null))
  <-> type_satisfied statics vars v (null_type is_null)
      ∨ (type_satisfied statics vars v (not_null_type non_null)
         ∧ match is_null with
           | Not_null => True
           | Maybe_null x => variable_is vars x 0
           end).
Proof.
  rewrite null_type_satisfied, not_null_type_satisfied.
  unfold value_head_type. rewrite type_satisfied_value. unfold descr_satisfied.
  rewrite head_of_kind_value_satisfied_eq. tauto.
Qed.

Section Value_head.

Variable join : join_scoped_envs.
Variable choices : meet_choices.
Variable mh : meet_expanded_head.
Variable v : value.

Hypothesis Hkeeps : meet_expanded_head_keeps mh.
Hypothesis Hsound : meet_expanded_head_sound mh.
Hypothesis Hnon_null :
  type_part_meet_ok non_null_value v
    (meet_head_of_kind_value_non_null join choices mh).

(* The side of the values that are not null. *)
Lemma meet_not_null_ok :
  type_part_meet_ok not_null_type v
    (meet_or_unknown_or_bottom
       (fun (env : Meet_env.t) (elt1 elt2 : head_of_kind_value_non_null) =>
          option_map
            (map_result
               (fun x : head_of_kind_value_non_null =>
                  @Ok head_of_kind_value_non_null x))
            (meet_head_of_kind_value_non_null join choices mh env elt1 elt2))).
Proof.
  intros menv a1 a2 result Henv H1 H2 Hmeet.
  refine (meet_or_unknown_or_bottom_ok_for _ _ (fits_as_type not_null_type)
            (satisfied_as_type not_null_type v) (fun a => a) _ menv a1 a2
            result _ _ Henv H1 H2 Hmeet).
  - intros head1 head2 result0 Hf1 Hf2 Hmeet0.
    destruct (meet_head_of_kind_value_non_null join choices mh menv head1
                head2)
      as [ result1 | ] eqn:Hmeet1; [ | discriminate Hmeet0 ].
    injection Hmeet0 as <-.
    apply (map_result_ok_for _ _ (fits_as_type non_null_value)
             (satisfied_as_type non_null_value v)
             (fits_as_type not_null_type) (satisfied_as_type not_null_type v)
             (fun x : head_of_kind_value_non_null =>
                @Ok head_of_kind_value_non_null x)
             menv head1 head2 result1).
    + intros tenv head H. exact H.
    + intros statics valuation head. reflexivity.
    + exact (Hnon_null menv head1 head2 result1 Henv Hf1 Hf2 Hmeet1).
  - intros statics valuation Hs. unfold satisfied_as_type in Hs.
    apply not_null_type_satisfied in Hs as [_ []].
Qed.

(* The side of null. *)
Definition meet_is_null (env : Meet_env.t) (is_null1 is_null2 : is_null)
    : option (Meet_result.t is_null) :=
  match is_null1, is_null2 with
  | Not_null, Not_null => Some (Meet_result.Bottom Both_inputs)
  | Not_null, Maybe_null _ => Some (Meet_result.Bottom Left_input)
  | Maybe_null _, Not_null => Some (Meet_result.Bottom Right_input)
  | Maybe_null is_null1, Maybe_null is_null2 =>
    option_map (map_result Maybe_null) (meet_relation mh env is_null1 is_null2)
  end.

Lemma meet_is_null_ok : type_part_meet_ok null_type v meet_is_null.
Proof.
  intros menv [ | x1 ] [ | x2 ] result Henv H1 H2 Hmeet; cbn in Hmeet.
  1-3: injection Hmeet as <-; apply meet_call_ok_bottom;
    intros statics valuation _ _ Hs1 Hs2; unfold satisfied_as_type in Hs1, Hs2;
    first [ apply null_type_satisfied in Hs1 as [_ []]
          | apply null_type_satisfied in Hs2 as [_ []] ].
  destruct (meet_relation mh menv x1 x2) as [ result0 | ] eqn:Hmeet0;
    [ | discriminate Hmeet ].
  injection Hmeet as <-.
  apply null_type_fits in H1. apply null_type_fits in H2.
  apply (map_result_ok_for _ _ relation_variable_fits
           (fun (statics : static_env) (valuation : env)
                (x : option variable) =>
              v = Value_const Null
              ∧ relation_variable_satisfied (fun _ => Some 1%Z) statics
                  valuation x)
           (fits_as_type null_type) (satisfied_as_type null_type v) Maybe_null
           menv x1 x2 result0).
  - intros tenv x H. apply null_type_fits. exact H.
  - intros statics valuation x. unfold satisfied_as_type.
    rewrite null_type_satisfied. unfold relation_variable_satisfied. split.
    + intros [Hnull Hx]. split; [ exact Hnull | ]. exists 1%Z. auto.
    + intros [Hnull (n & [= <-] & Hx)]. auto.
  - exact (part_meet_ok_fact _ _ (fun _ => v = Value_const Null) _
             (meet_relation_ok_for mh (fun _ => Some 1%Z) Hkeeps Hsound) menv x1
             x2 result0 Henv H1 H2 Hmeet0).
Qed.

End Value_head.

(* The meet of the side of null answers one of its inputs. *)
Lemma meet_is_null_answer (mh : meet_expanded_head) (menv menv' : Meet_env.t)
    (b1 b2 : is_null) (r : meet_return_value is_null) :
  meet_is_null mh menv b1 b2 = Some (Meet_result.Ok r menv') ->
  extract_value r b1 b2 = b1 ∨ extract_value r b1 b2 = b2.
Proof.
  destruct b1 as [ | x1 ], b2 as [ | x2 ]; cbn; try discriminate.
  unfold meet_relation. destruct x1 as [ x1 | ], x2 as [ x2 | ]; cbn.
  - destruct (Meet_env.add_alias menv (Var x1) (Var x2) mh)
      as [ [ menv1 | ] | ]; cbn; try discriminate.
    intros [= <- <-]. left. reflexivity.
  - intros [= <- <-]. left. reflexivity.
  - intros [= <- <-]. right. reflexivity.
  - intros [= <- <-]. left. reflexivity.
Qed.

(* The statement of the two alternatives says, of the answer, that one
   side is true. The meaning of a head says more of a value that is not
   null: that the variable for [Is_null] holds 0, which is of the other
   side. That is had from what the answer of that side can be
   ([meet_disjunction_answer_for]): the part of an input, which held 0
   before and holds it still, the environment of the meet having known
   it; or the bottom of that side, which has no variable. *)
Lemma meet_head_of_kind_value_ok_for :
  for_validity meet_head_of_kind_value_ok_statement.
Proof.
  intros join choices mh v Hjoin Hdefine Hkeeps Hsound Hnon_null menv
    [non_null1 is_null1] [non_null2 is_null2] result Henv H1 H2 Hmeet.
  unfold fits_as_type in H1, H2.
  apply value_head_fits in H1 as [Ha1 Hb1].
  apply value_head_fits in H2 as [Ha2 Hb2].
  unfold meet_head_of_kind_value in Hmeet.
  match type of Hmeet with
  | option_map _ ?m = _ => destruct m as [ result0 | ] eqn:Hmeet0
  end; [ | discriminate Hmeet ].
  injection Hmeet as <-.
  set (rebuild :=
    fun result : or_unknown_or_bottom head_of_kind_value_non_null * is_null
                 * variant_extensions =>
      let '(non_null, is_null, _) := result in
      Head_of_kind_value non_null is_null).
  set (x1 := (non_null1, is_null1, No_extensions)).
  set (x2 := (non_null2, is_null2, No_extensions)).
  assert (Hexclusive :
    ∀ (statics : static_env) (valuation : env)
      (a : or_unknown_or_bottom head_of_kind_value_non_null) (b : is_null),
      satisfied_as_type not_null_type v statics valuation a ->
      satisfied_as_type null_type v statics valuation b ->
      False).
  { intros statics valuation a b Hsa Hsb.
    apply not_null_type_satisfied in Hsa as [Hnot_null _].
    apply null_type_satisfied in Hsb as [Hnull _]. exact (Hnot_null Hnull). }
  assert (Hbottom_a :
    ∀ tenv : Typing_env.t, fits_as_type not_null_type tenv Bottom).
  { intros tenv. apply type_fits_lists. cbn. apply lists_fit_nil. }
  assert (Hbottom_b :
    ∀ tenv : Typing_env.t, fits_as_type null_type tenv Not_null).
  { intros tenv. apply type_fits_lists. cbn. apply lists_fit_nil. }
  assert (Hok :=
    meet_disjunction_ok_for _ _ (fits_as_type not_null_type)
      (satisfied_as_type not_null_type v) (fits_as_type null_type)
      (satisfied_as_type null_type v) join choices mh _ (meet_is_null mh)
      (fun _ => Bottom) (fun _ => Not_null) Hjoin Hdefine Hkeeps Hsound
      (type_part_stable not_null_type v) (type_part_stable null_type v)
      (fits_as_type_by_kinds not_null_type) (fits_as_type_by_kinds null_type)
      (meet_not_null_ok join choices mh v Hnon_null)
      (meet_is_null_ok mh v Hkeeps Hsound) Hexclusive Hbottom_a Hbottom_b menv
      x1 x2 result0 Henv (conj Ha1 (conj Hb1 (conj I I)))
      (conj Ha2 (conj Hb2 (conj I I))) Hmeet0).
  destruct Hok as [Hkeeps0 Hsound0]. split.
  - intros r' menv' Hresult.
    destruct (map_result_ok_cases _ result0 r' menv' Hresult)
      as (r & -> & ->).
    destruct (Hkeeps0 r menv' eq_refl)
      as (Hextends & Henv' & Hanswer & Hnames).
    split; [ exact Hextends | ]. split; [ exact Henv' | ].
    split; [ | exact Hnames ].
    change
      (fits_as_type value_head_type (Meet_env.typing_env menv')
         (extract_value (map_return_value rebuild r) (rebuild x1)
            (rebuild x2))).
    rewrite extract_value_map.
    destruct (extract_value r x1 x2) as [[a' b'] e'].
    destruct Hanswer as (Ha & Hb & _). apply value_head_fits.
    exact (conj Ha Hb).
  - intros statics valuation Hstatics Hvaluation Hs1 Hs2.
    unfold satisfied_as_type in Hs1, Hs2.
    apply value_head_satisfied in Hs1, Hs2.
    assert (Hd1 :
      disjunction_satisfied (satisfied_as_type not_null_type v)
        (satisfied_as_type null_type v) statics valuation x1).
    { destruct Hs1 as [ Hb | [Ha _] ]; [ right | left ]; split;
        first [ exact I | assumption ]. }
    assert (Hd2 :
      disjunction_satisfied (satisfied_as_type not_null_type v)
        (satisfied_as_type null_type v) statics valuation x2).
    { destruct Hs2 as [ Hb | [Ha _] ]; [ right | left ]; split;
        first [ exact I | assumption ]. }
    destruct (Hsound0 statics valuation Hstatics Hvaluation Hd1 Hd2)
      as (r & menv' & valuation' & -> & Hextended & Hvaluation' & Hanswer
          & Hvalued).
    exists (map_return_value rebuild r), menv', valuation'.
    split; [ apply map_result_ok_eq | ]. split; [ exact Hextended | ].
    split; [ exact Hvaluation' | ]. split; [ | exact Hvalued ].
    change
      (satisfied_as_type value_head_type v statics valuation'
         (extract_value (map_return_value rebuild r) (rebuild x1)
            (rebuild x2))).
    rewrite extract_value_map.
    assert (Hnames := extract_value_answer_for _ r x1 x2).
    destruct (extract_value r x1 x2) as [[a' b'] e'].
    destruct (meet_disjunction_answer_for _ _ join choices mh _ _ _ _ menv menv'
                non_null1 non_null2 a' is_null1 is_null2 b' No_extensions
                No_extensions e' r Hmeet0 Hnames)
      as [_ Hside].
    apply value_head_satisfied.
    destruct Hanswer as [ [Ha _] | [Hb _] ]; [ right | left; exact Hb ].
    split; [ exact Ha | ].
    assert (Hnot_null : v <> Value_const Null).
    { apply not_null_type_satisfied in Ha as [Hnot_null _]. exact Hnot_null. }
    assert (Hz1 :
      match is_null1 with
      | Not_null => True
      | Maybe_null x => variable_is valuation x 0
      end).
    { destruct Hs1 as [ Hb | [_ Hz] ]; [ | exact Hz ].
      apply null_type_satisfied in Hb as [Hnull _]. contradiction. }
    assert (Hz2 :
      match is_null2 with
      | Not_null => True
      | Maybe_null x => variable_is valuation x 0
      end).
    { destruct Hs2 as [ Hb | [_ Hz] ]; [ | exact Hz ].
      apply null_type_satisfied in Hb as [Hnull _]. contradiction. }
    assert (Hstable :
      ∀ b : is_null,
        type_fits (Meet_env.typing_env menv) (null_type b) ->
        match b with
        | Not_null => True
        | Maybe_null x => variable_is valuation x 0
        end ->
        match b with
        | Not_null => True
        | Maybe_null x => variable_is valuation' x 0
        end).
    { intros [ | x ] Hfit Hz; [ exact I | ]. apply null_type_fits in Hfit.
      destruct (proj2 (relation_variable_part_stable (fun _ => Some 0%Z))
                  statics (Meet_env.typing_env menv) valuation valuation' x
                  Hfit Hextended)
        as (n & [= <-] & Hx); [ | exact Hx ].
      exists 0%Z. split; [ reflexivity | exact Hz ]. }
    destruct Hside
      as [ -> | [ -> | [ -> | (menv0 & menv0' & rb & Hmeet_b & ->) ] ] ].
    + exact (Hstable is_null1 Hb1 Hz1).
    + exact (Hstable is_null2 Hb2 Hz2).
    + exact I.
    + destruct (meet_is_null_answer mh menv0 menv0' is_null1 is_null2 rb
                  Hmeet_b)
        as [ -> | -> ];
        [ exact (Hstable _ Hb1 Hz1) | exact (Hstable _ Hb2 Hz2) ].
Qed.

Lemma meet_value_heads_ok_for :
  for_validity meet_value_heads_ok_statement.
Proof.
  intros join choices mh v Hjoin Hdefine Hkeeps Hsound Hblocks Hclosures.
  apply (meet_head_of_kind_value_ok_for join choices mh v Hjoin Hdefine Hkeeps
           Hsound).
  apply (meet_head_of_kind_value_non_null_ok_for join choices mh v Hkeeps
    Hsound);
    [ | exact Hclosures ].
  exact (meet_variant_ok_for join choices mh v Hjoin Hdefine Hkeeps Hsound
           Hblocks).
Qed.

End Validity.

Theorem meet_relation_ok :
  meet_relation_ok_statement.
Proof.
  first
    [ exact (@meet_relation_ok_for standard_validity)
    | exact meet_relation_ok_for ].
Qed.

Theorem meet_disjunction_answer :
  meet_disjunction_answer_statement.
Proof.
  first
    [ exact (@meet_disjunction_answer_for standard_validity)
    | exact meet_disjunction_answer_for ].
Qed.

Theorem meet_head_of_kind_value_non_null_ok :
  meet_head_of_kind_value_non_null_ok_statement.
Proof.
  first
    [ exact (@meet_head_of_kind_value_non_null_ok_for standard_validity)
    | exact meet_head_of_kind_value_non_null_ok_for ].
Qed.

Theorem meet_variant_ok :
  meet_variant_ok_statement.
Proof.
  first
    [ exact (@meet_variant_ok_for standard_validity)
    | exact meet_variant_ok_for ].
Qed.

Theorem meet_head_of_kind_value_ok :
  meet_head_of_kind_value_ok_statement.
Proof.
  first
    [ exact (@meet_head_of_kind_value_ok_for standard_validity)
    | exact meet_head_of_kind_value_ok_for ].
Qed.

Theorem meet_value_heads_ok :
  meet_value_heads_ok_statement.
Proof.
  first
    [ exact (@meet_value_heads_ok_for standard_validity)
    | exact meet_value_heads_ok_for ].
Qed.

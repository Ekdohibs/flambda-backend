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


(* Proofs of the statements of FlambdaMeetTerminationSpec.v. Nothing
   here needs review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar FlambdaBindingTime
  FlambdaEnvAliases FlambdaEnvAliasesSpec FlambdaTypingEnv FlambdaMeetGrammar
  FlambdaExpandHead FlambdaMeetEnv FlambdaMeet FlambdaMeetSpec
  FlambdaMeetTerminationSpec.
From OxCamlFlambda2Types Require FlambdaMeetExamplesSpec
  FlambdaEnvAliasesProofs FlambdaTypingEnvInvariantSpec
  FlambdaTypingEnvInvariantProofs.
Import ListNotations.

(* ======================================================================== *)
(* Answers that may be a fatal error                                        *)
(* ======================================================================== *)

(* Where the first answers, the second gives the same answer. *)
Definition option_le {A : Type} (o o' : option A) : Prop :=
  ∀ a : A, o = Some a -> o' = Some a.

Lemma option_le_refl {A : Type} (o : option A) : option_le o o.
Proof. intros a Ho. exact Ho. Qed.

Lemma option_le_none {A : Type} (o : option A) : option_le None o.
Proof. intros a Ho. discriminate. Qed.

Lemma option_le_cases {A : Type} (o o' : option A) :
  option_le o o' <-> o = None ∨ o = o'.
Proof.
  split.
  - intros Hle. destruct o as [a | ]; [ right | left; reflexivity ].
    symmetry. exact (Hle a eq_refl).
  - intros [-> | ->]; [ apply option_le_none | apply option_le_refl ].
Qed.

Lemma option_le_bind {A B : Type} (m m' : option A) (k k' : A -> option B) :
  option_le m m' ->
  (∀ a : A, option_le (k a) (k' a)) ->
  option_le (match m with Some a => k a | None => None end)
    (match m' with Some a => k' a | None => None end).
Proof.
  intros Hm Hk b. destruct m as [a | ]; [ | discriminate ].
  rewrite (Hm a eq_refl). apply Hk.
Qed.

Lemma option_le_map {A B : Type} (f : A -> B) (m m' : option A) :
  option_le m m' -> option_le (option_map f m) (option_map f m').
Proof.
  intros Hm b. destruct m as [a | ]; [ | discriminate ].
  rewrite (Hm a eq_refl). intros Hb. exact Hb.
Qed.

(* A fold whose state may fail, and stays failed: if each step fails
   or is the step of a second fold, the fold fails or is the second
   fold. *)
Lemma fold_left_fails {S K : Type} (failed : S -> Prop) (g : S -> K -> S) :
  (∀ (s : S) (k : K), failed s -> failed (g s k)) ->
  ∀ (keys : list K) (s : S), failed s -> failed (List.fold_left g keys s).
Proof.
  intros Hfails keys. induction keys as [ | k keys IH]; intros s Hs.
  - exact Hs.
  - simpl. apply IH, Hfails, Hs.
Qed.

Lemma fold_left_le {S K : Type} (failed : S -> Prop) (g g' : S -> K -> S) :
  (∀ (s : S) (k : K), failed s -> failed (g s k)) ->
  (∀ (s : S) (k : K), failed (g s k) ∨ g s k = g' s k) ->
  ∀ (keys : list K) (s : S),
    failed (List.fold_left g keys s)
    ∨ List.fold_left g keys s = List.fold_left g' keys s.
Proof.
  intros Hfails Hstep keys. induction keys as [ | k keys IH]; intros s.
  - right. reflexivity.
  - simpl. destruct (Hstep s k) as [Hfailed | Heq].
    + left. exact (fold_left_fails failed g Hfails keys _ Hfailed).
    + destruct (IH (g s k)) as [Hfailed | Hrest];
        [ left; exact Hfailed | right ].
      rewrite Hrest, Heq. reflexivity.
Qed.

(* The steps of a proof that a function answers the same with a meet
   that answers more: the two sides are the same expression, with the
   one meet and with the other. *)
Ltac mono_step :=
  match goal with
  | |- option_le ?x ?x => apply option_le_refl
  | |- option_le None _ => apply option_le_none
  | H : meet_head_le ?mh ?mh' |- option_le (?mh _ _ _) (?mh' _ _ _) =>
    exact (H _ _ _)
  | |- option_le (option_map _ _) (option_map _ _) => apply option_le_map
  | |- option_le (match ?m with Some _ => _ | None => None end)
         (match ?m' with Some _ => _ | None => None end) =>
    apply option_le_bind; [ | intros ? ]
  | |- option_le (if ?b then _ else _) (if ?b then _ else _) => destruct b
  | |- option_le (match ?d with _ => _ end) (match ?d with _ => _ end) =>
    destruct d
  end.

Create HintDb mono.

Ltac mono := repeat first [ mono_step | solve [ eauto with mono ] ].

(* ======================================================================== *)
(* The functions of [Meet_env]                                              *)
(* ======================================================================== *)

Section Meet_env.

Variable mh mh' : meet_expanded_head.
Hypothesis Hle : meet_head_le mh mh'.

Lemma add_concrete_equation_on_canonical_name_mono (raise_on_bottom : bool)
    (env : Meet_env.t) (n : name) (ty : Expanded_type.t) :
  option_le
    (Meet_env.add_concrete_equation_on_canonical_name raise_on_bottom env n ty
       mh)
    (Meet_env.add_concrete_equation_on_canonical_name raise_on_bottom env n ty
       mh').
Proof. unfold Meet_env.add_concrete_equation_on_canonical_name. mono. Qed.

Lemma adding_equation_for_name_mono (env : Meet_env.t) (n : name)
    (f f' : Meet_env.t -> option (Or_bottom.t Meet_env.t)) :
  (∀ env : Meet_env.t, option_le (f env) (f' env)) ->
  option_le (Meet_env.adding_equation_for_name env n f)
    (Meet_env.adding_equation_for_name env n f').
Proof. intros Hf. unfold Meet_env.adding_equation_for_name. mono. Qed.

Lemma add_or_delay_concrete_equation_on_canonical_name_mono
    (raise_on_bottom : bool) (env : Meet_env.t) (n : name)
    (ty : Expanded_type.t) :
  option_le
    (Meet_env.add_or_delay_concrete_equation_on_canonical_name raise_on_bottom
       env n ty mh)
    (Meet_env.add_or_delay_concrete_equation_on_canonical_name raise_on_bottom
       env n ty mh').
Proof.
  unfold Meet_env.add_or_delay_concrete_equation_on_canonical_name. mono.
  apply adding_equation_for_name_mono. intros env'.
  apply add_concrete_equation_on_canonical_name_mono.
Qed.

Lemma add_concrete_equation_on_const_mono (raise_on_bottom : bool)
    (env : Meet_env.t) (c : const) (ty : Expanded_type.t) :
  option_le
    (Meet_env.add_concrete_equation_on_const raise_on_bottom env c ty mh)
    (Meet_env.add_concrete_equation_on_const raise_on_bottom env c ty mh').
Proof. unfold Meet_env.add_concrete_equation_on_const. mono. Qed.

Hint Resolve add_or_delay_concrete_equation_on_canonical_name_mono
  add_concrete_equation_on_const_mono : mono.

Lemma add_or_delay_concrete_equation_on_canonical_mono
    (raise_on_bottom : bool) (env : Meet_env.t) (s : simple)
    (ty : Expanded_type.t) :
  option_le
    (Meet_env.add_or_delay_concrete_equation_on_canonical raise_on_bottom env
       s ty mh)
    (Meet_env.add_or_delay_concrete_equation_on_canonical raise_on_bottom env
       s ty mh').
Proof. unfold Meet_env.add_or_delay_concrete_equation_on_canonical. mono. Qed.

Hint Resolve add_or_delay_concrete_equation_on_canonical_mono : mono.

Lemma add_concrete_equation_on_simple_mono (raise_on_bottom : bool)
    (env : Meet_env.t) (s : simple) (ty : Expanded_type.t) :
  option_le
    (Meet_env.add_concrete_equation_on_simple raise_on_bottom env s ty mh)
    (Meet_env.add_concrete_equation_on_simple raise_on_bottom env s ty mh').
Proof. unfold Meet_env.add_concrete_equation_on_simple. mono. Qed.

Lemma record_demotion_mono (raise_on_bottom : bool) (env : Meet_env.t)
    (kind : flambda_kind) (demoted : name) (canonical : simple) :
  option_le
    (Meet_env.record_demotion raise_on_bottom env kind demoted canonical mh)
    (Meet_env.record_demotion raise_on_bottom env kind demoted canonical mh').
Proof. unfold Meet_env.record_demotion. mono. Qed.

Hint Resolve add_concrete_equation_on_simple_mono record_demotion_mono : mono.

Lemma add_alias_between_canonicals_mono (raise_on_bottom : bool)
    (env : Meet_env.t) (kind : flambda_kind) (c1 c2 : simple) :
  option_le
    (Meet_env.add_alias_between_canonicals raise_on_bottom env kind c1 c2 mh)
    (Meet_env.add_alias_between_canonicals raise_on_bottom env kind c1 c2 mh').
Proof. unfold Meet_env.add_alias_between_canonicals. mono. Qed.

Hint Resolve add_alias_between_canonicals_mono : mono.

Lemma add_equation_on_simple0_mono (raise_on_bottom : bool)
    (env : Meet_env.t) (s : simple) (ty : flambda_type) :
  option_le (Meet_env.add_equation_on_simple0 raise_on_bottom env s ty mh)
    (Meet_env.add_equation_on_simple0 raise_on_bottom env s ty mh').
Proof. unfold Meet_env.add_equation_on_simple0. mono. Qed.

Hint Resolve add_equation_on_simple0_mono : mono.

Lemma add_equation0_mono (raise_on_bottom : bool) (env : Meet_env.t)
    (n : name) (ty : flambda_type) :
  option_le (Meet_env.add_equation0 raise_on_bottom env n ty mh)
    (Meet_env.add_equation0 raise_on_bottom env n ty mh').
Proof. unfold Meet_env.add_equation0. mono. Qed.

Hint Resolve add_equation0_mono : mono.

Lemma add_equations0_mono (raise_on_bottom : bool)
    (equations : list (name * flambda_type)) :
  ∀ env : Meet_env.t,
    option_le (Meet_env.add_equations0 raise_on_bottom env equations mh)
      (Meet_env.add_equations0 raise_on_bottom env equations mh').
Proof.
  induction equations as [ | [n ty] equations IH]; intros env; simpl; mono.
Qed.

Hint Resolve add_equations0_mono : mono.

Lemma add_env_extension0_mono (raise_on_bottom : bool) (env : Meet_env.t)
    (extension : env_extension) :
  option_le (Meet_env.add_env_extension0 raise_on_bottom env extension mh)
    (Meet_env.add_env_extension0 raise_on_bottom env extension mh').
Proof. unfold Meet_env.add_env_extension0. mono. Qed.

Hint Resolve add_env_extension0_mono : mono.

Lemma add_equation_strict_mono (env : Meet_env.t) (n : name)
    (ty : flambda_type) :
  option_le (Meet_env.add_equation_strict env n ty mh)
    (Meet_env.add_equation_strict env n ty mh').
Proof. unfold Meet_env.add_equation_strict. mono. Qed.

Lemma add_env_extension_strict_mono (env : Meet_env.t)
    (extension : env_extension) :
  option_le (Meet_env.add_env_extension_strict env extension mh)
    (Meet_env.add_env_extension_strict env extension mh').
Proof. unfold Meet_env.add_env_extension_strict. mono. Qed.

Lemma bottom_on_exception_mono (env : Meet_env.t)
    (result result' : option (Or_bottom.t Meet_env.t)) :
  option_le result result' ->
  option_le (Meet_env.bottom_on_exception env result)
    (Meet_env.bottom_on_exception env result').
Proof. intros Hresult. unfold Meet_env.bottom_on_exception. mono. Qed.

Lemma add_env_extension_maybe_bottom_mono (env : Meet_env.t)
    (extension : env_extension) :
  option_le (Meet_env.add_env_extension_maybe_bottom env extension mh)
    (Meet_env.add_env_extension_maybe_bottom env extension mh').
Proof. unfold Meet_env.add_env_extension_maybe_bottom. mono. Qed.

Lemma add_equation_mono (env : Meet_env.t) (n : name) (ty : flambda_type) :
  option_le (Meet_env.add_equation env n ty mh)
    (Meet_env.add_equation env n ty mh').
Proof.
  unfold Meet_env.add_equation. apply bottom_on_exception_mono. mono.
Qed.

Lemma add_equation_on_simple_mono (env : Meet_env.t) (s : simple)
    (ty : flambda_type) :
  option_le (Meet_env.add_equation_on_simple env s ty mh)
    (Meet_env.add_equation_on_simple env s ty mh').
Proof.
  unfold Meet_env.add_equation_on_simple. apply bottom_on_exception_mono.
  mono.
Qed.

Lemma add_env_extension_mono (env : Meet_env.t) (extension : env_extension) :
  option_le (Meet_env.add_env_extension env extension mh)
    (Meet_env.add_env_extension env extension mh').
Proof.
  unfold Meet_env.add_env_extension. apply bottom_on_exception_mono. mono.
Qed.

Hint Resolve add_equation_mono : mono.

Lemma add_equations_on_params_mono
    (params : list (variable * flambda_kind)) :
  ∀ (param_types : list flambda_type) (env : Meet_env.t),
    option_le (Meet_env.add_equations_on_params env params param_types mh)
      (Meet_env.add_equations_on_params env params param_types mh').
Proof.
  induction params as [ | param params IH]; intros [ | ty param_types] env;
    simpl; mono.
Qed.

Lemma add_alias0_mono (raise_on_bottom : bool) (env : Meet_env.t)
    (s1 s2 : simple) :
  option_le (Meet_env.add_alias0 raise_on_bottom env s1 s2 mh)
    (Meet_env.add_alias0 raise_on_bottom env s1 s2 mh').
Proof. unfold Meet_env.add_alias0. mono. Qed.

Lemma add_alias_mono (env : Meet_env.t) (s1 s2 : simple) :
  option_le (Meet_env.add_alias env s1 s2 mh)
    (Meet_env.add_alias env s1 s2 mh').
Proof. unfold Meet_env.add_alias. apply add_alias0_mono. Qed.

Lemma meet_env_meet_mono (env : Meet_env.t) (t1 t2 : flambda_type) :
  option_le (Meet_env.meet env t1 t2 mh) (Meet_env.meet env t1 t2 mh').
Proof. unfold Meet_env.meet. mono. Qed.

Lemma meet_type_mono (env : Meet_env.t) (t1 t2 : flambda_type) :
  option_le (Meet_env.meet_type env t1 t2 mh)
    (Meet_env.meet_type env t1 t2 mh').
Proof.
  unfold Meet_env.meet_type. apply option_le_map, meet_env_meet_mono.
Qed.

End Meet_env.

(* ======================================================================== *)
(* The functions that are given a meet for their parts                      *)
(* ======================================================================== *)

(* A fold whose state is an answer that may be a fatal error. *)
Lemma fold_left_option_le {S K : Type} (g g' : option S -> K -> option S)
    (keys : list K) :
  (∀ k : K, g None k = None) ->
  (∀ (s : option S) (k : K), option_le (g s k) (g' s k)) ->
  ∀ init : option S,
    option_le (List.fold_left g keys init) (List.fold_left g' keys init).
Proof.
  intros Hnone Hstep init. apply option_le_cases.
  apply (fold_left_le (fun s => s = None) g g').
  - intros s k ->. apply Hnone.
  - intros s k. apply option_le_cases, Hstep.
Qed.

(* What [meet_mapping] asks of the fold that it is given: it answers
   the same with a step that answers more. *)
Definition fold2_monotone {K A M : Type}
    (fold2 :
      (K -> option A -> option A -> mapping_state K A -> mapping_state K A) ->
      M -> M -> mapping_state K A -> mapping_state K A) : Prop :=
  ∀ (step step' :
       K -> option A -> option A -> mapping_state K A -> mapping_state K A)
    (left right : M) (init : mapping_state K A),
    (∀ (k : K) (a b : option A), step k a b None = None) ->
    (∀ (k : K) (a b : option A) (s : mapping_state K A),
      option_le (step k a b s) (step' k a b s)) ->
    option_le (fold2 step left right init) (fold2 step' left right init).

Lemma meet_mapping_mono {K A M : Type}
    (meet_data meet_data' : Meet_env.t -> A -> A -> option (Meet_result.t A))
    (fold2 :
      (K -> option A -> option A -> mapping_state K A -> mapping_state K A) ->
      M -> M -> mapping_state K A -> mapping_state K A)
    (env : Meet_env.t) (left right : M) (rebuild : list (K * A) -> M) :
  (∀ (env : Meet_env.t) (a b : A),
    option_le (meet_data env a b) (meet_data' env a b)) ->
  fold2_monotone fold2 ->
  option_le (meet_mapping meet_data fold2 env left right rebuild)
    (meet_mapping meet_data' fold2 env left right rebuild).
Proof.
  intros Hdata Hfold2. unfold meet_mapping. cbv zeta.
  apply option_le_bind; [ | intros state; apply option_le_refl ].
  apply Hfold2.
  - intros k a b. reflexivity.
  - intros k a b [[acc | ] | ]; mono.
Qed.

Lemma map_meet_mono {K A : Type} (eqb : K -> K -> bool)
    (meet_data meet_data' : Meet_env.t -> A -> A -> option (Meet_result.t A))
    (env : Meet_env.t) (left right : Keyed.finite_map K A) :
  (∀ (env : Meet_env.t) (a b : A),
    option_le (meet_data env a b) (meet_data' env a b)) ->
  option_le (map_meet eqb meet_data env left right)
    (map_meet eqb meet_data' env left right).
Proof.
  intros Hdata. unfold map_meet. cbv zeta.
  apply meet_mapping_mono; [ exact Hdata | ].
  intros step step' m1 m2 init Hnone Hstep. unfold Keyed.fold2_on.
  apply fold_left_option_le.
  - intros k. apply Hnone.
  - intros s k. apply Hstep.
Qed.

Lemma combine_results2_mono {A B C : Type} (env : Meet_env.t)
    (meet_a meet_a' : Meet_env.t -> A -> A -> option (Meet_result.t A))
    (meet_b meet_b' : Meet_env.t -> B -> B -> option (Meet_result.t B))
    (left_a right_a : A) (left_b right_b : B) (rebuild : A -> B -> C) :
  (∀ (env : Meet_env.t) (a a' : A),
    option_le (meet_a env a a') (meet_a' env a a')) ->
  (∀ (env : Meet_env.t) (b b' : B),
    option_le (meet_b env b b') (meet_b' env b b')) ->
  option_le
    (combine_results2 env meet_a meet_b left_a right_a left_b right_b rebuild)
    (combine_results2 env meet_a' meet_b' left_a right_a left_b right_b
       rebuild).
Proof. intros Ha Hb. unfold combine_results2. mono. Qed.

Lemma combine_results3_mono {A B C D : Type} (env : Meet_env.t)
    (meet_a meet_a' : Meet_env.t -> A -> A -> option (Meet_result.t A))
    (meet_b meet_b' : Meet_env.t -> B -> B -> option (Meet_result.t B))
    (meet_c meet_c' : Meet_env.t -> C -> C -> option (Meet_result.t C))
    (left_a right_a : A) (left_b right_b : B) (left_c right_c : C)
    (rebuild : A -> B -> C -> D) :
  (∀ (env : Meet_env.t) (a a' : A),
    option_le (meet_a env a a') (meet_a' env a a')) ->
  (∀ (env : Meet_env.t) (b b' : B),
    option_le (meet_b env b b') (meet_b' env b b')) ->
  (∀ (env : Meet_env.t) (c c' : C),
    option_le (meet_c env c c') (meet_c' env c c')) ->
  option_le
    (combine_results3 env meet_a meet_b meet_c left_a right_a left_b right_b
       left_c right_c rebuild)
    (combine_results3 env meet_a' meet_b' meet_c' left_a right_a left_b
       right_b left_c right_c rebuild).
Proof. intros Ha Hb Hc. unfold combine_results3. mono. Qed.

Lemma meet_unknown_mono {A : Type}
    (meet_contents meet_contents' :
      Meet_env.t -> A -> A -> option (Meet_result.t A))
    (contents_is_bottom : A -> bool) (env : Meet_env.t)
    (or_unknown1 or_unknown2 : Or_unknown.t A) :
  (∀ (env : Meet_env.t) (a a' : A),
    option_le (meet_contents env a a') (meet_contents' env a a')) ->
  option_le
    (meet_unknown meet_contents contents_is_bottom env or_unknown1
       or_unknown2)
    (meet_unknown meet_contents' contents_is_bottom env or_unknown1
       or_unknown2).
Proof. intros Hcontents. unfold meet_unknown. mono. Qed.

Lemma meet_or_unknown_or_bottom_mono {A B : Type}
    (meet_elt meet_elt' : Meet_env.t -> A -> A -> option (Meet_result.t B))
    (env : Meet_env.t) (input1 input2 : or_unknown_or_bottom A) :
  (∀ (env : Meet_env.t) (a a' : A),
    option_le (meet_elt env a a') (meet_elt' env a a')) ->
  option_le (meet_or_unknown_or_bottom meet_elt env input1 input2)
    (meet_or_unknown_or_bottom meet_elt' env input1 input2).
Proof. intros Helt. unfold meet_or_unknown_or_bottom. mono. Qed.

(* A merge that threads a state and may stop with a fatal error. *)
Lemma merge_fold_failing_mono {K A B C S : Type} (eqb : K -> K -> bool)
    (keys : list K)
    (f f' : K -> option A -> option B -> S -> option (option C * S))
    (m1 : Keyed.finite_map K A) (m2 : Keyed.finite_map K B) (init : S) :
  (∀ (k : K) (a : option A) (b : option B) (s : S),
    option_le (f k a b s) (f' k a b s)) ->
  option_le (merge_fold_failing eqb keys f m1 m2 init)
    (merge_fold_failing eqb keys f' m1 m2 init).
Proof.
  intros Hf. unfold merge_fold_failing, Keyed.merge_fold_on.
  match goal with
  | |- option_le (let '(_, _) := List.fold_left ?g _ ?init in _)
         (let '(_, _) := List.fold_left ?g' _ _ in _) =>
    destruct (fold_left_le (fun acc => snd acc = None) g g') with
      (keys := keys) (s := init) as [Hfailed | Heq]
  end.
  - intros acc k Hacc. rewrite Hacc. reflexivity.
  - intros acc k. destruct (snd acc) as [state | ]; [ | left; reflexivity ].
    destruct (proj1 (option_le_cases _ _)
                (Hf k (Keyed.lookup eqb k m1) (Keyed.lookup eqb k m2) state))
      as [Hnone | Hsame].
    + left. rewrite Hnone. reflexivity.
    + right. rewrite Hsame. reflexivity.
  - destruct (List.fold_left _ keys _) as [out state]. simpl in Hfailed.
    subst state. apply option_le_none.
  - rewrite Heq. apply option_le_refl.
Qed.

(* What [meet_row_like] asks of the merge that it is given: it answers
   the same with a function for the cases that answers more. *)
Definition merge_monotone {row_tag case known : Type}
    (merge :
      (row_tag -> option (Or_unknown.t case) -> option (Or_unknown.t case) ->
       row_like_state ->
       option (option (Or_unknown.t case) * row_like_state)) ->
      known -> known -> row_like_state -> option (known * row_like_state))
    : Prop :=
  ∀ (f f' :
       row_tag -> option (Or_unknown.t case) -> option (Or_unknown.t case) ->
       row_like_state -> option (option (Or_unknown.t case) * row_like_state))
    (known1 known2 : known) (state : row_like_state),
    (∀ (tag : row_tag) (case1 case2 : option (Or_unknown.t case))
       (state : row_like_state),
      option_le (f tag case1 case2 state) (f' tag case1 case2 state)) ->
    option_le (merge f known1 known2 state) (merge f' known1 known2 state).

(* The functions above, as steps. *)
Ltac mono_parts :=
  match goal with
  | |- option_le (combine_results2 _ _ _ _ _ _ _ _)
         (combine_results2 _ _ _ _ _ _ _ _) =>
    apply combine_results2_mono; intros
  | |- option_le (combine_results3 _ _ _ _ _ _ _ _ _ _ _)
         (combine_results3 _ _ _ _ _ _ _ _ _ _ _) =>
    apply combine_results3_mono; intros
  | |- option_le (meet_unknown _ _ _ _ _) (meet_unknown _ _ _ _ _) =>
    apply meet_unknown_mono; intros
  | |- option_le (meet_or_unknown_or_bottom _ _ _ _)
         (meet_or_unknown_or_bottom _ _ _ _) =>
    apply meet_or_unknown_or_bottom_mono; intros
  | |- option_le (map_meet _ _ _ _ _) (map_meet _ _ _ _ _) =>
    apply map_meet_mono; intros
  end.

(* ======================================================================== *)
(* The meet, given the meet of less fuel                                    *)
(* ======================================================================== *)

Section Step.

Variable join : join_scoped_envs.
Variable choices : meet_choices.
Variable mh mh' : meet_expanded_head.
Hypothesis Hjoin : join_monotone join.
Hypothesis Hle : meet_head_le mh mh'.

Lemma join_mono (cut_after : Scope.t) (initial_env : Meet_env.t)
    (initial_tenv : Typing_env.t) (scoped_envs : list Typing_env.t) :
  option_le (join mh cut_after initial_env initial_tenv scoped_envs)
    (join mh' cut_after initial_env initial_tenv scoped_envs).
Proof.
  intros result.
  exact (Hjoin mh mh' cut_after initial_env result initial_tenv scoped_envs
           Hle).
Qed.

Lemma meet_mono (env : Meet_env.t) (t1 t2 : flambda_type) :
  option_le (FlambdaMeet.meet mh env t1 t2) (FlambdaMeet.meet mh' env t1 t2).
Proof. apply meet_type_mono, Hle. Qed.

Lemma env_extension_strict_mono (env : Meet_env.t)
    (extension : env_extension) :
  option_le (Meet_env.add_env_extension_strict env extension mh)
    (Meet_env.add_env_extension_strict env extension mh').
Proof. apply add_env_extension_strict_mono, Hle. Qed.

Lemma equation_strict_mono (env : Meet_env.t) (n : name)
    (ty : flambda_type) :
  option_le (Meet_env.add_equation_strict env n ty mh)
    (Meet_env.add_equation_strict env n ty mh').
Proof. apply add_equation_strict_mono, Hle. Qed.

Hint Resolve join_mono meet_mono env_extension_strict_mono
  equation_strict_mono : mono.

Lemma meet_disjunction_mono {A B : Type}
    (meet_a meet_a' : Meet_env.t -> A -> A -> option (Meet_result.t A))
    (meet_b meet_b' : Meet_env.t -> B -> B -> option (Meet_result.t B))
    (bottom_a : unit -> A) (bottom_b : unit -> B) (initial_env : Meet_env.t)
    (val_a1 : A) (val_b1 : B) (extensions1 : variant_extensions)
    (val_a2 : A) (val_b2 : B) (extensions2 : variant_extensions) :
  (∀ (env : Meet_env.t) (a a' : A),
    option_le (meet_a env a a') (meet_a' env a a')) ->
  (∀ (env : Meet_env.t) (b b' : B),
    option_le (meet_b env b b') (meet_b' env b b')) ->
  option_le
    (meet_disjunction join choices mh meet_a meet_b bottom_a bottom_b
       initial_env val_a1 val_b1 extensions1 val_a2 val_b2 extensions2)
    (meet_disjunction join choices mh' meet_a' meet_b' bottom_a bottom_b
       initial_env val_a1 val_b1 extensions1 val_a2 val_b2 extensions2).
Proof.
  intros Ha Hb. unfold meet_disjunction. cbv zeta. mono.
Qed.

Lemma meet_row_like_mono {lattice shape maps_to row_tag known : Type}
    (meet_maps_to meet_maps_to' :
      Meet_env.t -> maps_to -> maps_to -> option (Meet_result.t maps_to))
    (equal_index subset_index : lattice -> lattice -> bool)
    (union_index : lattice -> lattice -> lattice)
    (meet_shape : shape -> shape -> Or_bottom.t shape)
    (is_empty_map_known : known -> bool)
    (get_singleton_map_known :
      known ->
      option (row_tag * row_like_case lattice shape maps_to env_extension))
    (merge_map_known :
      (row_tag ->
       option
         (Or_unknown.t (row_like_case lattice shape maps_to env_extension)) ->
       option
         (Or_unknown.t (row_like_case lattice shape maps_to env_extension)) ->
       row_like_state ->
       option
         (option
            (Or_unknown.t
               (row_like_case lattice shape maps_to env_extension))
          * row_like_state)) ->
      known -> known -> row_like_state -> option (known * row_like_state))
    (initial_env : Meet_env.t) (known1 known2 : known)
    (other1 other2 :
      Or_bottom.t (row_like_case lattice shape maps_to env_extension)) :
  (∀ (env : Meet_env.t) (a b : maps_to),
    option_le (meet_maps_to env a b) (meet_maps_to' env a b)) ->
  merge_monotone merge_map_known ->
  option_le
    (meet_row_like join choices mh meet_maps_to equal_index subset_index
       union_index meet_shape is_empty_map_known get_singleton_map_known
       merge_map_known initial_env known1 known2 other1 other2)
    (meet_row_like join choices mh' meet_maps_to' equal_index subset_index
       union_index meet_shape is_empty_map_known get_singleton_map_known
       merge_map_known initial_env known1 known2 other1 other2).
Proof.
  intros Hmaps Hmerge. unfold meet_row_like. cbv zeta.
  destruct (Meet_env.enter_scope initial_env)
    as [[common_scope initial_tenv] base_env].
  apply option_le_bind.
  - apply Hmerge. intros tag case1 case2 state. mono.
  - intros [merged state]. mono.
Qed.

Lemma alias_mono (env : Meet_env.t) (s1 s2 : simple) :
  option_le (Meet_env.add_alias env s1 s2 mh)
    (Meet_env.add_alias env s1 s2 mh').
Proof. apply add_alias_mono, Hle. Qed.

Hint Resolve alias_mono : mono.

Lemma meet_relation_mono (env : Meet_env.t) (var1 var2 : option variable) :
  option_le (meet_relation mh env var1 var2)
    (meet_relation mh' env var1 var2).
Proof. unfold meet_relation. mono. Qed.

Lemma meet_array_of_types_mono (env : Meet_env.t)
    (fields1 fields2 : list flambda_type) (length : nat) :
  option_le (meet_array_of_types mh env fields1 fields2 length)
    (meet_array_of_types mh' env fields1 fields2 length).
Proof.
  unfold meet_array_of_types. cbv zeta.
  apply meet_mapping_mono; [ intros; apply meet_mono | ].
  intros step step' left right init Hnone Hstep.
  apply fold_left_option_le.
  - intros k. apply Hnone.
  - intros s k. apply Hstep.
Qed.

Lemma meet_int_indexed_product_mono (env : Meet_env.t)
    (fields1 fields2 : list flambda_type) :
  option_le (meet_int_indexed_product mh env fields1 fields2)
    (meet_int_indexed_product mh' env fields1 fields2).
Proof.
  unfold meet_int_indexed_product. apply meet_array_of_types_mono.
Qed.

Lemma meet_product_function_slot_indexed_mono (env : Meet_env.t)
    (components1 components2 : finite_map flambda_type) :
  option_le
    (meet_product_function_slot_indexed mh env components1 components2)
    (meet_product_function_slot_indexed mh' env components1 components2).
Proof.
  unfold meet_product_function_slot_indexed. apply map_meet_mono.
  intros. apply meet_mono.
Qed.

Lemma meet_product_value_slot_indexed_mono (env : Meet_env.t)
    (components1 components2 : finite_map flambda_type) :
  option_le (meet_product_value_slot_indexed mh env components1 components2)
    (meet_product_value_slot_indexed mh' env components1 components2).
Proof.
  unfold meet_product_value_slot_indexed. apply map_meet_mono.
  intros. apply meet_mono.
Qed.

Hint Resolve meet_relation_mono meet_int_indexed_product_mono
  meet_product_function_slot_indexed_mono
  meet_product_value_slot_indexed_mono : mono.

Ltac mono' :=
  repeat first [ mono_step | mono_parts | solve [ eauto with mono ] ].

Lemma meet_closures_entry_mono (env : Meet_env.t)
    (entry1 entry2 : closures_entry) :
  option_le (meet_closures_entry mh env entry1 entry2)
    (meet_closures_entry mh' env entry1 entry2).
Proof. unfold meet_closures_entry. mono'. Qed.

Hint Resolve meet_closures_entry_mono : mono.

Lemma meet_row_like_for_blocks_mono (env : Meet_env.t)
    (blocks1 blocks2 : row_like_for_blocks) :
  option_le (meet_row_like_for_blocks join choices mh env blocks1 blocks2)
    (meet_row_like_for_blocks join choices mh' env blocks1 blocks2).
Proof.
  unfold meet_row_like_for_blocks.
  destruct blocks1 as [known1 other1 alloc_mode1].
  destruct blocks2 as [known2 other2 alloc_mode2]. cbv zeta.
  apply combine_results2_mono; [ | intros; apply option_le_refl ].
  intros env' left right. apply meet_row_like_mono.
  - intros. apply meet_int_indexed_product_mono.
  - intros f f' k1 k2 state Hf. apply merge_fold_failing_mono. exact Hf.
Qed.

Lemma meet_row_like_for_closures_mono (env : Meet_env.t)
    (closures1 closures2 : row_like_for_closures) :
  option_le
    (meet_row_like_for_closures join choices mh env closures1 closures2)
    (meet_row_like_for_closures join choices mh' env closures1 closures2).
Proof.
  unfold meet_row_like_for_closures.
  destruct closures1 as [known1]. destruct closures2 as [known2]. cbv zeta.
  apply option_le_bind; [ | intros result; apply option_le_refl ].
  apply meet_row_like_mono.
  - intros. apply meet_closures_entry_mono.
  - intros f f' k1 k2 state Hf. apply merge_fold_failing_mono.
    intros k a b s. mono.
Qed.

Lemma reduce_inverse_relations_mono (env : Meet_env.t)
    (naked_immediates : Or_unknown.t (list Z))
    (inverse_relations : list (Relation.t * name)) :
  option_le
    (reduce_inverse_relations choices mh env naked_immediates
       inverse_relations)
    (reduce_inverse_relations choices mh' env naked_immediates
       inverse_relations).
Proof.
  unfold reduce_inverse_relations.
  destruct naked_immediates as [imms | ]; [ | apply option_le_refl ].
  cbv zeta. apply fold_left_option_le.
  - intros k. reflexivity.
  - intros [[env' | ] | ] inverse; mono.
Qed.

Hint Resolve reduce_inverse_relations_mono : mono.

Lemma meet_head_of_kind_naked_immediate_mono (env : Meet_env.t)
    (t1 t2 : head_of_kind_naked_immediate) :
  option_le (meet_head_of_kind_naked_immediate choices mh env t1 t2)
    (meet_head_of_kind_naked_immediate choices mh' env t1 t2).
Proof. unfold meet_head_of_kind_naked_immediate. cbv zeta. mono. Qed.

Hint Resolve meet_row_like_for_blocks_mono meet_row_like_for_closures_mono
  meet_head_of_kind_naked_immediate_mono : mono.

Lemma meet_variant_mono (env : Meet_env.t)
    (is_int1 get_tag1 : option variable)
    (blocks1 : Or_unknown.t row_like_for_blocks)
    (imms1 : Or_unknown.t flambda_type) (is_int2 get_tag2 : option variable)
    (blocks2 : Or_unknown.t row_like_for_blocks)
    (imms2 : Or_unknown.t flambda_type)
    (extensions1 extensions2 : variant_extensions) :
  option_le
    (meet_variant join choices mh env is_int1 get_tag1 blocks1 imms1 is_int2
       get_tag2 blocks2 imms2 extensions1 extensions2)
    (meet_variant join choices mh' env is_int1 get_tag1 blocks1 imms1 is_int2
       get_tag2 blocks2 imms2 extensions1 extensions2).
Proof.
  unfold meet_variant. cbv zeta.
  apply combine_results2_mono; [ intros; apply meet_relation_mono | ].
  intros env' [[[g1 b1] i1] e1] [[[g2 b2] i2] e2].
  apply option_le_map, meet_disjunction_mono; intros; mono'.
Qed.

Hint Resolve meet_variant_mono : mono.

Lemma meet_head_of_kind_value_non_null_mono (env : Meet_env.t)
    (head1 head2 : head_of_kind_value_non_null) :
  option_le (meet_head_of_kind_value_non_null join choices mh env head1 head2)
    (meet_head_of_kind_value_non_null join choices mh' env head1 head2).
Proof.
  unfold meet_head_of_kind_value_non_null. destruct head1, head2; mono'.
Qed.

Hint Resolve meet_head_of_kind_value_non_null_mono : mono.

Lemma meet_head_of_kind_value_mono (env : Meet_env.t)
    (head1 head2 : head_of_kind_value) :
  option_le (meet_head_of_kind_value join choices mh env head1 head2)
    (meet_head_of_kind_value join choices mh' env head1 head2).
Proof.
  unfold meet_head_of_kind_value.
  destruct head1 as [non_null1 is_null1].
  destruct head2 as [non_null2 is_null2].
  cbv zeta. apply option_le_map, meet_disjunction_mono; intros; mono'.
Qed.

Hint Resolve meet_head_of_kind_value_mono : mono.

Lemma meet_expanded_head0_mono (env : Meet_env.t)
    (descr1 descr2 : Expanded_type.head) :
  option_le (meet_expanded_head0 join choices mh env descr1 descr2)
    (meet_expanded_head0 join choices mh' env descr1 descr2).
Proof. unfold meet_expanded_head0. destruct descr1, descr2; mono'. Qed.

Lemma meet_expanded_head_step_mono :
  meet_head_le (meet_expanded_head_step join choices mh)
    (meet_expanded_head_step join choices mh').
Proof.
  intros env e1 e2. unfold meet_expanded_head_step.
  apply meet_or_unknown_or_bottom_mono. intros.
  apply meet_expanded_head0_mono.
Qed.

End Step.

(* ======================================================================== *)
(* More fuel changes no answer                                              *)
(* ======================================================================== *)

Theorem forgetful_join_monotone : forgetful_join_monotone_statement.
Proof.
  intros mh mh' cut_after initial_env result initial_tenv scoped_envs _
    Hresult.
  exact Hresult.
Qed.

Theorem meet_expanded_head_step_monotone :
  meet_expanded_head_step_monotone_statement.
Proof.
  intros join choices mh mh' Hjoin Hle.
  exact (meet_expanded_head_step_mono join choices mh mh' Hjoin Hle).
Qed.

Lemma meet_head_le_refl (mh : meet_expanded_head) : meet_head_le mh mh.
Proof. intros env e1 e2 result Hresult. exact Hresult. Qed.

Lemma meet_head_le_trans (mh mh' mh'' : meet_expanded_head) :
  meet_head_le mh mh' -> meet_head_le mh' mh'' -> meet_head_le mh mh''.
Proof.
  intros Hfirst Hsecond env e1 e2 result Hresult.
  exact (Hsecond env e1 e2 result (Hfirst env e1 e2 result Hresult)).
Qed.

Lemma meet_expanded_head_succ (join : join_scoped_envs)
    (choices : meet_choices) (fuel : nat) :
  join_monotone join ->
  meet_head_le (Meet.meet_expanded_head join choices fuel)
    (Meet.meet_expanded_head join choices (S fuel)).
Proof.
  intros Hjoin. induction fuel as [ | fuel IH].
  - intros env e1 e2 result Hresult. discriminate.
  - exact (meet_expanded_head_step_mono join choices _ _ Hjoin IH).
Qed.

Theorem meet_expanded_head_fuel_monotone :
  meet_expanded_head_fuel_monotone_statement.
Proof.
  intros join choices fuel fuel' Hjoin Hfuel.
  induction Hfuel as [ | fuel' Hfuel IH]; [ apply meet_head_le_refl | ].
  exact (meet_head_le_trans _ _ _ IH
           (meet_expanded_head_succ join choices fuel' Hjoin)).
Qed.

Section Fuel.

Variable join : join_scoped_envs.
Variable choices : meet_choices.
Variable fuel fuel' : nat.
Hypothesis Hjoin : join_monotone join.
Hypothesis Hfuel : fuel <= fuel'.

Let Hle :
  meet_head_le (Meet.meet_expanded_head join choices fuel)
    (Meet.meet_expanded_head join choices fuel') :=
  meet_expanded_head_fuel_monotone join choices fuel fuel' Hjoin Hfuel.

Lemma meet_fuel_mono (tenv : Typing_env.t) (t1 t2 : flambda_type) :
  option_le (Meet.meet join choices fuel tenv t1 t2)
    (Meet.meet join choices fuel' tenv t1 t2).
Proof.
  unfold Meet.meet. mono. apply meet_type_mono. exact Hle.
Qed.

Lemma meet_shape_fuel_mono (tenv : Typing_env.t) (t shape : flambda_type) :
  option_le (Meet.meet_shape join choices fuel tenv t shape)
    (Meet.meet_shape join choices fuel' tenv t shape).
Proof.
  unfold Meet.meet_shape. mono. apply meet_fuel_mono.
Qed.

Lemma flambda2_types_add_equation_mono (tenv : Typing_env.t) (n : name)
    (ty : flambda_type) :
  option_le (Flambda2_types.add_equation join choices fuel tenv n ty)
    (Flambda2_types.add_equation join choices fuel' tenv n ty).
Proof.
  unfold Flambda2_types.add_equation, Meet_env.use_meet_env. cbv zeta.
  apply option_le_map, add_equation_mono. exact Hle.
Qed.

Lemma flambda2_types_add_env_extension_mono (tenv : Typing_env.t)
    (extension : env_extension) :
  option_le (Flambda2_types.add_env_extension join choices fuel tenv extension)
    (Flambda2_types.add_env_extension join choices fuel' tenv extension).
Proof.
  unfold Flambda2_types.add_env_extension, Meet_env.use_meet_env. cbv zeta.
  apply option_le_map, add_env_extension_mono. exact Hle.
Qed.

Lemma flambda2_types_add_equations_on_params_mono (tenv : Typing_env.t)
    (params : list (variable * flambda_kind))
    (param_types : list flambda_type) :
  option_le
    (Flambda2_types.add_equations_on_params join choices fuel tenv params
       param_types)
    (Flambda2_types.add_equations_on_params join choices fuel' tenv params
       param_types).
Proof.
  unfold Flambda2_types.add_equations_on_params, Meet_env.use_meet_env.
  cbv zeta. apply option_le_map, add_equations_on_params_mono. exact Hle.
Qed.

Lemma flambda2_types_add_relation_mono (tenv : Typing_env.t) (n : name)
    (scrutinee : simple) (name_ty : flambda_type)
    (scrutinee_head : variable -> head_of_kind_value) :
  option_le
    (Flambda2_types.add_relation join choices fuel tenv n scrutinee name_ty
       scrutinee_head)
    (Flambda2_types.add_relation join choices fuel' tenv n scrutinee name_ty
       scrutinee_head).
Proof.
  unfold Flambda2_types.add_relation, Meet_env.use_meet_env. cbv zeta.
  apply option_le_map, option_le_bind.
  - apply add_equation_mono. exact Hle.
  - intros env. destruct n as [var | sym]; [ | apply option_le_refl ].
    apply add_equation_on_simple_mono. exact Hle.
Qed.

End Fuel.

Theorem meet_fuel_monotone : meet_fuel_monotone_statement.
Proof.
  intros join choices fuel fuel' tenv t1 t2 result Hjoin Hfuel.
  exact (meet_fuel_mono join choices fuel fuel' Hjoin Hfuel tenv t1 t2 result).
Qed.

Theorem meet_shape_fuel_monotone : meet_shape_fuel_monotone_statement.
Proof.
  intros join choices fuel fuel' tenv t shape result Hjoin Hfuel.
  exact (meet_shape_fuel_mono join choices fuel fuel' Hjoin Hfuel tenv t shape
           result).
Qed.

Theorem typing_env_functions_fuel_monotone :
  typing_env_functions_fuel_monotone_statement.
Proof.
  intros join choices fuel fuel' tenv tenv' Hjoin Hfuel.
  split; [ | split; [ | split ] ].
  - intros n ty.
    exact (flambda2_types_add_equation_mono join choices fuel fuel' Hjoin
             Hfuel tenv n ty tenv').
  - intros extension.
    exact (flambda2_types_add_env_extension_mono join choices fuel fuel' Hjoin
             Hfuel tenv extension tenv').
  - intros params param_types.
    exact (flambda2_types_add_equations_on_params_mono join choices fuel fuel'
             Hjoin Hfuel tenv params param_types tenv').
  - intros n scrutinee name_ty scrutinee_head.
    exact (flambda2_types_add_relation_mono join choices fuel fuel' Hjoin
             Hfuel tenv n scrutinee name_ty scrutinee_head tenv').
Qed.

Theorem meet_answer_unique : meet_answer_unique_statement.
Proof.
  intros join choices fuel fuel' tenv t1 t2 result result' Hjoin Hresult
    Hresult'.
  pose proof (meet_fuel_monotone join choices fuel (Nat.max fuel fuel') tenv
                t1 t2 result Hjoin (Nat.le_max_l fuel fuel') Hresult)
    as Hmax.
  pose proof (meet_fuel_monotone join choices fuel' (Nat.max fuel fuel') tenv
                t1 t2 result' Hjoin (Nat.le_max_r fuel fuel') Hresult')
    as Hmax'.
  congruence.
Qed.

(* ======================================================================== *)
(* A fuel that is enough                                                    *)
(* ======================================================================== *)

Theorem meet_expanded_head_over_none : meet_expanded_head_over_none_statement.
Proof.
  intros join choices fuel. induction fuel as [ | fuel IH];
    [ reflexivity | ].
  simpl. rewrite IH. reflexivity.
Qed.

(* The meet over [mh0] with the fuel [n + k] is the meet with the fuel
   [n] over the meet over [mh0] with the fuel [k]. *)
Lemma meet_expanded_head_over_add (join : join_scoped_envs)
    (choices : meet_choices) (mh0 : meet_expanded_head) (n k : nat) :
  meet_expanded_head_over join choices mh0 (n + k)
  = meet_expanded_head_over join choices
      (meet_expanded_head_over join choices mh0 k) n.
Proof.
  induction n as [ | n IH]; [ reflexivity | ].
  simpl. rewrite IH. reflexivity.
Qed.

(* The meet with the fuel [n + k] is the meet with the fuel [n] over
   the meet with the fuel [k]. *)
Lemma meet_expanded_head_add (join : join_scoped_envs)
    (choices : meet_choices) (n k : nat) :
  Meet.meet_expanded_head join choices (n + k)
  = meet_expanded_head_over join choices
      (Meet.meet_expanded_head join choices k) n.
Proof.
  rewrite <- (meet_expanded_head_over_none join choices (n + k)),
    meet_expanded_head_over_add, meet_expanded_head_over_none.
  reflexivity.
Qed.

Theorem fuel_enough_stable : fuel_enough_stable_statement.
Proof.
  intros join choices fuel fuel' menv e1 e2 Henough Hfuel.
  replace fuel' with (fuel + (fuel' - fuel)) by lia.
  assert (Hanswer :
    Meet.meet_expanded_head join choices (fuel + (fuel' - fuel)) menv e1 e2
    = Meet.meet_expanded_head join choices fuel menv e1 e2).
  { rewrite meet_expanded_head_add. apply Henough. }
  split; [ | exact Hanswer ].
  intros mh0. rewrite meet_expanded_head_over_add, Hanswer. apply Henough.
Qed.

Theorem meet_fuel_enough_stable : meet_fuel_enough_stable_statement.
Proof.
  intros join choices fuel fuel' tenv t1 t2 Henough Hfuel.
  replace fuel' with (fuel + (fuel' - fuel)) by lia.
  assert (Hanswer :
    Meet_env.meet_type (Meet_env.create tenv) t1 t2
      (Meet.meet_expanded_head join choices (fuel + (fuel' - fuel)))
    = Meet_env.meet_type (Meet_env.create tenv) t1 t2
        (Meet.meet_expanded_head join choices fuel)).
  { rewrite meet_expanded_head_add. apply Henough. }
  split.
  - intros mh0. rewrite meet_expanded_head_over_add, Hanswer. apply Henough.
  - unfold Meet.meet. rewrite Hanswer. reflexivity.
Qed.

(* The meet answers no more than the meet over any meet. *)
Lemma meet_expanded_head_le_over (join : join_scoped_envs)
    (choices : meet_choices) (mh0 : meet_expanded_head) (fuel : nat) :
  join_monotone join ->
  meet_head_le (Meet.meet_expanded_head join choices fuel)
    (meet_expanded_head_over join choices mh0 fuel).
Proof.
  intros Hjoin. induction fuel as [ | fuel IH].
  - intros env e1 e2 result Hresult. discriminate.
  - exact (meet_expanded_head_step_mono join choices _ _ Hjoin IH).
Qed.

Theorem answer_fuel_enough : answer_fuel_enough_statement.
Proof.
  intros join choices fuel menv e1 e2 result Hjoin Hresult mh0.
  rewrite Hresult.
  exact (meet_expanded_head_le_over join choices mh0 fuel Hjoin menv e1 e2
           result Hresult).
Qed.

Theorem meet_answer_fuel_enough : meet_answer_fuel_enough_statement.
Proof.
  intros join choices fuel tenv t1 t2 result Hjoin Hbottom Hresult mh0.
  unfold Meet.meet in Hresult. rewrite Hbottom in Hresult.
  destruct (Meet_env.meet_type (Meet_env.create tenv) t1 t2
              (Meet.meet_expanded_head join choices fuel))
    as [answer | ] eqn:Hanswer; [ | discriminate ].
  exact (meet_type_mono _ _
           (meet_expanded_head_le_over join choices mh0 fuel Hjoin)
           (Meet_env.create tenv) t1 t2 answer Hanswer).
Qed.

(* ======================================================================== *)
(* Inputs for which a fuel is known                                         *)
(* ======================================================================== *)

(* The step of the meet of two flat expanded types does not depend on
   the meet that it is given. *)
Lemma flat_step_any (join : join_scoped_envs) (choices : meet_choices)
    (mh mh' : meet_expanded_head) (menv : Meet_env.t)
    (e1 e2 : Expanded_type.t) :
  flat e1 = true ->
  flat e2 = true ->
  meet_expanded_head_step join choices mh menv e1 e2
  = meet_expanded_head_step join choices mh' menv e1 e2.
Proof.
  unfold flat, meet_expanded_head_step, meet_or_unknown_or_bottom.
  destruct (Expanded_type.descr e1) as [ | head1 | ];
    destruct (Expanded_type.descr e2) as [ | head2 | ]; try reflexivity.
  intros Hflat1 Hflat2.
  destruct head1 as [v1 | [imms1 relations1] | k1 numbers1 | [] | []];
    destruct head2 as [v2 | [imms2 relations2] | k2 numbers2 | [] | []];
    try discriminate; try reflexivity.
  cbn in Hflat1, Hflat2.
  destruct relations1; [ | discriminate ].
  destruct relations2; [ | discriminate ].
  cbn.
  destruct imms1 as [imms1 | ]; destruct imms2 as [imms2 | ]; cbn;
    try reflexivity.
Qed.

Lemma unknown_or_bottom_step_any (join : join_scoped_envs)
    (choices : meet_choices) (mh mh' : meet_expanded_head) (menv : Meet_env.t)
    (e1 e2 : Expanded_type.t) :
  Expanded_type.is_unknown e1 || Expanded_type.is_bottom e1
  || Expanded_type.is_unknown e2 || Expanded_type.is_bottom e2
  = true ->
  meet_expanded_head_step join choices mh menv e1 e2
  = meet_expanded_head_step join choices mh' menv e1 e2.
Proof.
  unfold Expanded_type.is_unknown, Expanded_type.is_bottom,
    meet_expanded_head_step, meet_or_unknown_or_bottom.
  destruct (Expanded_type.descr e1) as [ | head1 | ];
    destruct (Expanded_type.descr e2) as [ | head2 | ]; try reflexivity.
  discriminate.
Qed.

Theorem flat_fuel_enough : flat_fuel_enough_statement.
Proof.
  intros join choices menv e1 e2 Hflat1 Hflat2 mh0.
  exact (flat_step_any join choices _ _ menv e1 e2 Hflat1 Hflat2).
Qed.

Theorem unknown_or_bottom_fuel_enough :
  unknown_or_bottom_fuel_enough_statement.
Proof.
  intros join choices menv e1 e2 Hone mh0.
  exact (unknown_or_bottom_step_any join choices _ _ menv e1 e2 Hone).
Qed.

(* A type that has a head is not an alias. *)
Lemma of_non_alias_type_not_alias (t : flambda_type) (e : Expanded_type.t) :
  Expanded_type.of_non_alias_type t = Some e -> get_alias_opt t = None.
Proof.
  destruct t as [ty | ty | k ty | ty | ty];
    destruct ty as [ | [h | s] | ]; cbn; try discriminate; reflexivity.
Qed.

(* The meet of two types that are not aliases is the meet of their
   heads. *)
Lemma meet_type_non_alias (menv : Meet_env.t) (t1 t2 : flambda_type)
    (e1 e2 : Expanded_type.t) (mh : meet_expanded_head) :
  Expanded_type.of_non_alias_type t1 = Some e1 ->
  Expanded_type.of_non_alias_type t2 = Some e2 ->
  Meet_env.meet_type menv t1 t2 mh
  = if negb (kind_eqb (kind t1) (kind t2))
    then None
    else option_map (map_result Expanded_type.to_type) (mh menv e1 e2).
Proof.
  intros He1 He2.
  unfold Meet_env.meet_type, Meet_env.meet,
    Typing_env.get_alias_then_canonical_simple_exn, expand_head0.
  rewrite (of_non_alias_type_not_alias t1 e1 He1),
    (of_non_alias_type_not_alias t2 e2 He2), He1, He2.
  destruct (negb (kind_eqb (kind t1) (kind t2))); reflexivity.
Qed.

Theorem non_alias_fuel_enough : non_alias_fuel_enough_statement.
Proof.
  intros join choices fuel tenv t1 t2 e1 e2 He1 He2 Henough mh0.
  rewrite !(meet_type_non_alias _ t1 t2 e1 e2 _ He1 He2), Henough.
  reflexivity.
Qed.

Theorem flat_types_fuel_enough : flat_types_fuel_enough_statement.
Proof.
  intros join choices tenv t1 t2 e1 e2 He1 He2 Hflat1 Hflat2.
  apply (non_alias_fuel_enough join choices 1 tenv t1 t2 e1 e2 He1 He2).
  apply flat_fuel_enough; assumption.
Qed.

(* The canonical element of a constant is the constant. *)
Lemma canonical_of_const (tenv : Typing_env.t) (c : const) :
  Typing_env.get_alias_then_canonical_simple_exn tenv
    (Some Name_mode.in_types) None (this_const c)
  = Some (Some (Const c)).
Proof.
  destruct c as [ | | | | | | | | | | | | | | k]; try reflexivity.
  destruct k as [ | [] | | ]; reflexivity.
Qed.

(* The meet of a type that is not an alias with the type of a constant
   is the meet of the head of the type with the head of the constant,
   and what [Meet_env] makes of its answer. *)
Lemma meet_type_const_right (menv : Meet_env.t) (t : flambda_type)
    (e : Expanded_type.t) (c : const) (mh mh' : meet_expanded_head) :
  Expanded_type.of_non_alias_type t = Some e ->
  mh menv e (Expanded_type.create_const c)
  = mh' menv e (Expanded_type.create_const c) ->
  Meet_env.meet_type menv t (this_const c) mh
  = Meet_env.meet_type menv t (this_const c) mh'.
Proof.
  intros He Hagree.
  unfold Meet_env.meet_type, Meet_env.meet.
  destruct (negb (kind_eqb (kind t) (kind (this_const c)))); [ reflexivity | ].
  rewrite canonical_of_const.
  unfold Typing_env.get_alias_then_canonical_simple_exn, expand_head0.
  rewrite (of_non_alias_type_not_alias t e He), He. cbn.
  unfold Meet_env.add_concrete_equation_on_const. rewrite Hagree.
  reflexivity.
Qed.

Lemma meet_type_const_left (menv : Meet_env.t) (t : flambda_type)
    (e : Expanded_type.t) (c : const) (mh mh' : meet_expanded_head) :
  Expanded_type.of_non_alias_type t = Some e ->
  mh menv e (Expanded_type.create_const c)
  = mh' menv e (Expanded_type.create_const c) ->
  Meet_env.meet_type menv (this_const c) t mh
  = Meet_env.meet_type menv (this_const c) t mh'.
Proof.
  intros He Hagree.
  unfold Meet_env.meet_type, Meet_env.meet.
  destruct (negb (kind_eqb (kind (this_const c)) (kind t))); [ reflexivity | ].
  rewrite canonical_of_const.
  unfold Typing_env.get_alias_then_canonical_simple_exn, expand_head0.
  rewrite (of_non_alias_type_not_alias t e He), He. cbn.
  unfold Meet_env.add_concrete_equation_on_const. rewrite Hagree.
  reflexivity.
Qed.

Theorem const_fuel_enough : const_fuel_enough_statement.
Proof.
  intros join choices fuel tenv t e c He Henough.
  split; intros mh0.
  - exact (meet_type_const_right _ t e c _ _ He (Henough mh0)).
  - exact (meet_type_const_left _ t e c _ _ He (Henough mh0)).
Qed.

(* Giving an environment other aliases changes no type. *)
Lemma find_with_aliases (tenv : Typing_env.t) (aliases : Aliases.t) (n : name)
    (kind : option flambda_kind) :
  Typing_env.find (Typing_env.with_aliases tenv aliases) n kind
  = Typing_env.find tenv n kind.
Proof. reflexivity. Qed.

(* The meet of an alias of a canonical name with the type of a constant
   asks one meet: that of the type of the name with the head of the
   constant. *)
Lemma meet_type_alias_const (menv : Meet_env.t) (t ty : flambda_type)
    (n : name) (e : Expanded_type.t) (c : const)
    (mh mh' : meet_expanded_head) :
  aliases_consistent (Typing_env.aliases (Meet_env.typing_env menv)) ->
  is_canonical (Typing_env.aliases (Meet_env.typing_env menv))
    (simple_of_name n) ->
  Typing_env.get_alias_then_canonical_simple_exn (Meet_env.typing_env menv)
    (Some Name_mode.in_types) None t
  = Some (Some (simple_of_name n)) ->
  Typing_env.find (Meet_env.typing_env menv) n (Some (kind t)) = Some ty ->
  Expanded_type.of_non_alias_type ty = Some e ->
  (∀ menv' : Meet_env.t,
    mh menv' e (Expanded_type.create_const c)
    = mh' menv' e (Expanded_type.create_const c)) ->
  Meet_env.meet_type menv t (this_const c) mh
  = Meet_env.meet_type menv t (this_const c) mh'.
Proof.
  intros Hconsistent Hcanonical Hsimple Hfind He Hagree.
  unfold Meet_env.meet_type, Meet_env.meet.
  destruct (negb (kind_eqb (kind t) (kind (this_const c)))); [ reflexivity | ].
  rewrite Hsimple, canonical_of_const.
  cbn beta iota.
  unfold Meet_env.add_alias_between_canonicals.
  assert (Hne : simple_equal (simple_of_name n) (Const c) = false)
    by (destruct n; reflexivity).
  rewrite Hne.
  unfold Typing_env.add_alias, Typing_env.aliases_add.
  destruct (FlambdaEnvAliasesProofs.add_name_constant _
              (Typing_env.binding_time_resolver (Meet_env.typing_env menv))
              (Typing_env.names_to_types (Meet_env.typing_env menv))
              (Typing_env.aliases (Meet_env.typing_env menv)) n c Hconsistent
              Hcanonical) as [[result [Hadd [Hdemoted Hcanon]]] _].
  rewrite Hadd.
  cbn [Typing_env.add_alias_result_t Typing_env.demoted_name
       Typing_env.canonical_element].
  rewrite Hdemoted, Hcanon.
  unfold Meet_env.record_demotion.
  cbn [Meet_env.typing_env Meet_env.with_typing_env Meet_env.map_typing_env].
  rewrite find_with_aliases, Hfind.
  cbn beta iota.
  unfold expand_head0.
  rewrite (of_non_alias_type_not_alias ty e He), He.
  cbn beta iota delta [Meet_env.add_or_delay_concrete_equation_on_canonical].
  unfold Meet_env.add_concrete_equation_on_const.
  rewrite Hagree.
  reflexivity.
Qed.

Theorem alias_const_fuel_enough : alias_const_fuel_enough_statement.
Proof.
  intros join choices fuel tenv t ty n e c Hconsistent Hcanonical Hsimple Hfind
    He Henough mh0.
  apply (meet_type_alias_const (Meet_env.create tenv) t ty n e c);
    try assumption.
  intros menv'. exact (Henough menv' mh0).
Qed.

Theorem alias_flat_const_fuel_enough : alias_flat_const_fuel_enough_statement.
Proof.
  intros join choices tenv t ty n e c Hconsistent Hcanonical Hsimple Hfind He
    Hflat Hconst.
  apply (alias_const_fuel_enough join choices 1 tenv t ty n e c);
    try assumption.
  intros menv. apply flat_fuel_enough; assumption.
Qed.

Theorem example_alias_const_fuel : example_alias_const_fuel_statement.
Proof.
  split; [ | split; [ eexists; vm_compute; reflexivity | reflexivity ] ].
  intros join choices.
  assert (Hwf :
    FlambdaTypingEnvInvariantSpec.typing_env_well_formed example_fuel_env
    = true) by (vm_compute; reflexivity).
  pose proof
    (proj1 (FlambdaTypingEnvInvariantProofs.typing_env_well_formed_iff _) Hwf)
    as Hconditions.
  destruct Hconditions as [_ [_ [Hconsistent _]]].
  apply (alias_flat_const_fuel_enough join choices example_fuel_env
           example_fuel_alias
           (Type_naked_immediate Unknown) (Name_var 1%positive)
           (Expanded_type.create_unknown (Naked_number Kind_naked_immediate))
           (Naked_immediate 0%Z) Hconsistent);
    reflexivity.
Qed.

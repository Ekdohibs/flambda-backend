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

(* Proofs of the statements of FlambdaMeetDepthSpec.v. Nothing here
   needs review.

   The step of the meet is taken function by function, for two meets
   of less fuel and a class of inputs for each side ([plain_type] with
   its two parameters). Of each function two things are shown together
   ([same_kept]): it answers the same with the two meets, given that
   the meets of types made of them answer the same on the types inside
   ([Hinside] of the section [Step]); and where it answers an
   environment, that has the typing environment of the one it was
   given. The second is what makes the first go through a disjunction
   and a row-like structure: the environments that are cut there have
   nothing in them, so no extension is added and no equation.

   The types with no name inside are the instance in which both sides
   have the class of one depth less, and the meets of types answer the
   same by induction on the depth ([meet_steady],
   [nameless_over_steady]). The meet with a shape is the instance in
   which one side has types inside that say nothing or have no value,
   and the meets of types answer the same because the meet of such a
   type with any other asks no meet of less fuel
   ([meet_of_nothing_steady], [meet_with_nothing_steady]).

   What a meet makes is followed the same way, function by function
   ([answers]: a value that an answer makes has a property; nothing is
   said of an answer that is one of the inputs). For the functions of
   a disjunction and of a row-like structure it rests on the second
   thing above: the extension that is recorded is cut from an
   environment to which nothing was added, so it is empty. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar FlambdaBindingTime
  FlambdaEnvAliases FlambdaTypingEnv FlambdaMeetGrammar FlambdaExpandHead
  FlambdaMeetEnv FlambdaMeet FlambdaMeetSpec FlambdaMeetTerminationSpec
  FlambdaMeetTerminationProofs FlambdaMeetDepthSpec.
From OxCamlFlambda2Types Require FlambdaMeetExamplesSpec FlambdaJoinEnv
  FlambdaJoinMeet FlambdaJoinNothingSpec FlambdaJoinNothingProofs.
Import ListNotations.

(* ======================================================================== *)
(* Types with no name inside                                                *)
(* ======================================================================== *)

Lemma forallb_more {A : Type} (f g : A -> bool) (l : list A) :
  (∀ a : A, f a = true -> g a = true) ->
  forallb f l = true -> forallb g l = true.
Proof.
  intros Hfg. rewrite !forallb_forall. intros Hall a Ha.
  apply Hfg, Hall, Ha.
Qed.

(* [plain_type] grows with what it asks of the types inside. *)
Lemma plain_type_more (relations : bool)
    (inside inside' : flambda_type -> bool) (ty : flambda_type) :
  (∀ ty : flambda_type, inside ty = true -> inside' ty = true) ->
  plain_type relations inside ty = true ->
  plain_type relations inside' ty = true.
Proof.
  intros Hinside.
  assert (Hcase :
    ∀ case : row_like_block_case,
      plain_block_case inside case = true ->
      plain_block_case inside' case = true).
  { intros case. unfold plain_block_case.
    rewrite !andb_true_iff. intros [Hfields Hextension].
    split; [ exact (forallb_more _ _ _ Hinside Hfields) | exact Hextension ]. }
  assert (Hblocks :
    ∀ blocks : row_like_for_blocks,
      plain_blocks inside blocks = true ->
      plain_blocks inside' blocks = true).
  { intros [known other alloc_mode]. unfold plain_blocks.
    rewrite !andb_true_iff. intros [Hknown Hother]. split.
    - revert Hknown. apply forallb_more. intros [tag [case | ]]; cbn;
        [ apply Hcase | trivial ].
    - destruct other as [case | ]; [ apply Hcase, Hother | reflexivity ]. }
  assert (Hclosures :
    ∀ closures : row_like_for_closures,
      plain_closures inside closures = true ->
      plain_closures inside' closures = true).
  { intros [known]. unfold plain_closures. apply forallb_more.
    intros [slot [[function_types closure_types value_slot_types] index
                   extension]].
    cbn. rewrite !andb_true_iff. intros [[Hclosure Hslots] Hextension].
    repeat split; try exact Hextension.
    - revert Hclosure. apply forallb_more. intros closure. apply Hinside.
    - revert Hslots. apply forallb_more. intros slot'. apply Hinside. }
  destruct ty as [ty | ty | k ty | ty | ty];
    destruct ty as [ | [head | s] | ]; cbn; try trivial.
  destruct head as [non_null is_null]. cbn.
  rewrite !andb_true_iff. intros [Hnon_null His_null].
  split; [ | exact His_null ].
  destruct non_null as [ | non_null | ]; try reflexivity.
  destruct non_null
    as [is_int immediates get_tag blocks extensions is_unique | alloc_mode
       | k contents alloc_mode | closures alloc_mode | strs];
    cbn in Hnon_null |- *; try reflexivity.
  - rewrite !andb_true_iff in Hnon_null |- *.
    destruct Hnon_null as [[[[His_int Hget_tag] Himmediates] Hblocks'] Hext].
    repeat split; try assumption.
    + destruct immediates as [ty | ]; [ | trivial ].
      apply orb_true_iff in Himmediates. apply orb_true_iff.
      destruct Himmediates as [Hty | Hbottom];
        [ left; apply Hinside, Hty | right; exact Hbottom ].
    + destruct blocks as [blocks | ]; [ apply Hblocks, Hblocks' | trivial ].
  - apply Hinside, Hnon_null.
  - apply Hclosures, Hnon_null.
Qed.

Lemma nameless_or_const_more (inside inside' : flambda_type -> bool)
    (ty : flambda_type) :
  (∀ ty : flambda_type, inside ty = true -> inside' ty = true) ->
  nameless_or_const inside ty = true -> nameless_or_const inside' ty = true.
Proof.
  intros Hinside. unfold nameless_or_const.
  destruct (get_alias_opt ty) as [[v | s | c] | ]; try discriminate;
    apply plain_type_more, Hinside.
Qed.

Lemma nameless_within_succ (depth : nat) (ty : flambda_type) :
  nameless_within depth ty = true -> nameless_within (S depth) ty = true.
Proof.
  revert ty. induction depth as [ | depth IH]; intros ty; [ discriminate | ].
  cbn [nameless_within]. apply nameless_or_const_more. exact IH.
Qed.

Theorem nameless_within_more : nameless_within_more_statement.
Proof.
  intros depth depth' ty Hle. induction Hle as [ | depth' Hle IH];
    [ trivial | ].
  intros Hty. apply nameless_within_succ, IH, Hty.
Qed.

Theorem flat_nameless : flat_nameless_statement.
Proof.
  intros [kind descr]. unfold flat, nameless_expanded_within,
    Expanded_type.to_type. cbn [Expanded_type.descr Expanded_type.kind].
  destruct descr as [ | head | ].
  - intros _. destruct kind as [ | [] | | ]; reflexivity.
  - destruct head as [v | [imms relations] | k numbers | [] | []]; cbn;
      try discriminate; trivial.
  - intros _. destruct kind as [ | [] | | ]; reflexivity.
Qed.

Theorem const_nameless : const_nameless_statement.
Proof.
  intros ty c Halias. cbn [nameless_within]. unfold nameless_or_const.
  rewrite Halias.
  destruct c as [ | | | | | | | | | | | | | | k]; try (split; reflexivity).
  destruct k as [ | [] | | ]; split; reflexivity.
Qed.

Theorem forgetful_join_of_nothing : forgetful_join_of_nothing_statement.
Proof.
  intros env scoped_envs _ _. exists env.
  split; [ reflexivity | intros mh; reflexivity ].
Qed.

Theorem example_model_join : example_model_join_statement.
Proof.
  assert (Hnone : example_model_join_answer (example_join_fuel 0) = None)
    by (vm_compute; reflexivity).
  split; [ vm_compute; reflexivity | split; [ | split; [ exact Hnone | ] ] ].
  - intros Hsame.
    pose proof (f_equal (option_map Meet_env.names_to_avoid) Hsame) as Hnames.
    vm_compute in Hnames. discriminate Hnames.
  - intros Hjoin.
    destruct
      (Hjoin (Meet_env.create example_join_env)
         [Typing_env.increment_scope example_join_env])
      as (result & _ & Hresult).
    + discriminate.
    + intros scoped [<- | []]. reflexivity.
    + specialize
        (Hresult
           (Meet.meet_expanded_head FlambdaMeetExamplesSpec.forgetful_join
              repaired_choices 3)).
      change (example_model_join_answer (example_join_fuel 0) = Some result)
        in Hresult.
      rewrite Hnone in Hresult. discriminate Hresult.
Qed.

Theorem model_join_of_nothing : model_join_of_nothing_statement.
Proof.
  intros join_choices join_depth fuel Hfuel env scoped_envs Hnonempty Hall.
  exact
    (FlambdaJoinNothingProofs.model_join_scoped_envs_of_nothing join_choices
       join_depth fuel env scoped_envs Hfuel Hnonempty Hall).
Qed.

(* ======================================================================== *)
(* The same answer, with the typing environment that was given              *)
(* ======================================================================== *)

(* Where an answer has an environment, its typing environment is that
   of [env]. The names to avoid, which are the model's own, and the
   names that an equation is being added to, may be others. *)
Definition kept {A : Type} (env : Meet_env.t) (o : option (Meet_result.t A))
    : Prop :=
  ∀ (r : meet_return_value A) (env' : Meet_env.t),
    o = Some (Meet_result.Ok r env') ->
    Meet_env.typing_env env' = Meet_env.typing_env env.

(* Two answers are the same, and have the typing environment of [env]
   if any. *)
Definition same_kept {A : Type} (env : Meet_env.t)
    (o o' : option (Meet_result.t A)) : Prop :=
  o = o' ∧ kept env o.

(* Two meets answer so on the inputs that have [P]. *)
Definition steady {A B : Type} (P1 P2 : A -> Prop)
    (m m' : Meet_env.t -> A -> A -> option (Meet_result.t B)) : Prop :=
  ∀ (env : Meet_env.t) (a b : A),
    P1 a -> P2 b -> same_kept env (m env a b) (m' env a b).

Lemma same_kept_none {A : Type} (env : Meet_env.t) :
  same_kept env (@None (Meet_result.t A)) None.
Proof. split; [ reflexivity | intros r env'; discriminate ]. Qed.

Lemma same_kept_bottom {A : Type} (env : Meet_env.t)
    (r : meet_return_value unit) :
  same_kept env (Some (@Meet_result.Bottom A r)) (Some (Meet_result.Bottom r)).
Proof. split; [ reflexivity | intros r' env'; discriminate ]. Qed.

Lemma same_kept_ok {A : Type} (env : Meet_env.t) (r : meet_return_value A) :
  same_kept env (Some (Meet_result.Ok r env)) (Some (Meet_result.Ok r env)).
Proof.
  split; [ reflexivity | intros r' env' [= _ <-]; reflexivity ].
Qed.

Lemma same_kept_ok_at {A : Type} (env env1 : Meet_env.t)
    (r : meet_return_value A) :
  Meet_env.typing_env env1 = Meet_env.typing_env env ->
  same_kept env (Some (Meet_result.Ok r env1)) (Some (Meet_result.Ok r env1)).
Proof.
  intros Henv1. split; [ reflexivity | intros r' env' [= _ <-]; exact Henv1 ].
Qed.

(* What is kept of an environment is kept of one with the same typing
   environment. *)
Lemma same_kept_at {A : Type} (env env1 : Meet_env.t)
    (o o' : option (Meet_result.t A)) :
  Meet_env.typing_env env1 = Meet_env.typing_env env ->
  same_kept env1 o o' -> same_kept env o o'.
Proof.
  intros Henv1 [Hsame Hkept]. split; [ exact Hsame | ].
  intros r env' Ho. rewrite <- Henv1. exact (Hkept r env' Ho).
Qed.

(* An answer that is the same and kept is, by cases, no answer, bottom,
   or an answer with the typing environment given. *)
Lemma same_kept_cases {A : Type} (env : Meet_env.t)
    (o o' : option (Meet_result.t A)) :
  same_kept env o o' ->
  o' = o
  ∧ (o = None
     ∨ (∃ r : meet_return_value unit, o = Some (Meet_result.Bottom r))
     ∨ ∃ (r : meet_return_value A) (env1 : Meet_env.t),
         Meet_env.typing_env env1 = Meet_env.typing_env env
         ∧ o = Some (Meet_result.Ok r env1)).
Proof.
  intros [<- Hkept]. split; [ reflexivity | ].
  destruct o as [[r | r env'] | ]; [ right; left | right; right | left ].
  - exists r. reflexivity.
  - exists r, env'. split; [ exact (Hkept r env' eq_refl) | reflexivity ].
  - reflexivity.
Qed.

Lemma map_result_ok {A B : Type} (f : A -> B) (result : Meet_result.t A)
    (r : meet_return_value B) (env : Meet_env.t) :
  map_result f result = Meet_result.Ok r env ->
  ∃ r0 : meet_return_value A, result = Meet_result.Ok r0 env.
Proof.
  destruct result as [r0 | [ | | | a] env0]; cbn; intros [= ]; subst;
    eexists; reflexivity.
Qed.

Lemma same_kept_map {A B : Type} (f : A -> B) (env : Meet_env.t)
    (o o' : option (Meet_result.t A)) :
  same_kept env o o' ->
  same_kept env (option_map (map_result f) o) (option_map (map_result f) o').
Proof.
  intros [<- Hkept]. split; [ reflexivity | ].
  intros r env' Hmap. destruct o as [result | ]; [ | discriminate ].
  injection Hmap as Hmap. apply map_result_ok in Hmap.
  destruct Hmap as [r0 ->]. exact (Hkept r0 env' eq_refl).
Qed.

(* The cases of [same_kept_cases], as a tactic: the answer of the
   second meet is replaced by that of the first, which is then taken
   by cases. In the last case the environment answered is [env1], and
   [Henv1] says that it has the typing environment given. *)
Ltac kept_cases H env1 Henv1 :=
  let Heq := fresh "Heq" in
  let Hcases := fresh "Hcases" in
  let r := fresh "r" in
  apply same_kept_cases in H; destruct H as [Heq Hcases];
  rewrite Heq; clear Heq;
  destruct Hcases as [-> | [[r ->] | (r & env1 & Henv1 & ->)]].

(* ======================================================================== *)
(* The functions that are given a meet for their parts                      *)
(* ======================================================================== *)

Lemma combine_results2_steady {A B C : Type} (PA1 PA2 : A -> Prop)
    (PB1 PB2 : B -> Prop)
    (meet_a meet_a' : Meet_env.t -> A -> A -> option (Meet_result.t A))
    (meet_b meet_b' : Meet_env.t -> B -> B -> option (Meet_result.t B))
    (env : Meet_env.t) (left_a right_a : A) (left_b right_b : B)
    (rebuild : A -> B -> C) :
  steady PA1 PA2 meet_a meet_a' ->
  steady PB1 PB2 meet_b meet_b' ->
  PA1 left_a -> PA2 right_a -> PB1 left_b -> PB2 right_b ->
  same_kept env
    (combine_results2 env meet_a meet_b left_a right_a left_b right_b rebuild)
    (combine_results2 env meet_a' meet_b' left_a right_a left_b right_b
       rebuild).
Proof.
  intros Ha Hb Hla Hra Hlb Hrb. unfold combine_results2.
  pose proof (Ha env left_a right_a Hla Hra) as Hfirst.
  kept_cases Hfirst env1 Henv1;
    [ apply same_kept_none | apply same_kept_bottom | ].
  apply (same_kept_at env env1 _ _ Henv1).
  pose proof (Hb env1 left_b right_b Hlb Hrb) as Hsecond.
  kept_cases Hsecond env2 Henv2;
    [ apply same_kept_none | apply same_kept_bottom
    | apply same_kept_ok_at, Henv2 ].
Qed.

Lemma combine_results3_steady {A B C D : Type} (PA1 PA2 : A -> Prop)
    (PB1 PB2 : B -> Prop) (PC1 PC2 : C -> Prop)
    (meet_a meet_a' : Meet_env.t -> A -> A -> option (Meet_result.t A))
    (meet_b meet_b' : Meet_env.t -> B -> B -> option (Meet_result.t B))
    (meet_c meet_c' : Meet_env.t -> C -> C -> option (Meet_result.t C))
    (env : Meet_env.t) (left_a right_a : A) (left_b right_b : B)
    (left_c right_c : C) (rebuild : A -> B -> C -> D) :
  steady PA1 PA2 meet_a meet_a' ->
  steady PB1 PB2 meet_b meet_b' ->
  steady PC1 PC2 meet_c meet_c' ->
  PA1 left_a -> PA2 right_a -> PB1 left_b -> PB2 right_b ->
  PC1 left_c -> PC2 right_c ->
  same_kept env
    (combine_results3 env meet_a meet_b meet_c left_a right_a left_b right_b
       left_c right_c rebuild)
    (combine_results3 env meet_a' meet_b' meet_c' left_a right_a left_b
       right_b left_c right_c rebuild).
Proof.
  intros Ha Hb Hc Hla Hra Hlb Hrb Hlc Hrc. unfold combine_results3.
  pose proof (Ha env left_a right_a Hla Hra) as Hfirst.
  kept_cases Hfirst env1 Henv1;
    [ apply same_kept_none | apply same_kept_bottom | ].
  apply (same_kept_at env env1 _ _ Henv1).
  pose proof (Hb env1 left_b right_b Hlb Hrb) as Hsecond.
  kept_cases Hsecond env2 Henv2;
    [ apply same_kept_none | apply same_kept_bottom | ].
  apply (same_kept_at env1 env2 _ _ Henv2).
  pose proof (Hc env2 left_c right_c Hlc Hrc) as Hthird.
  kept_cases Hthird env3 Henv3;
    [ apply same_kept_none | apply same_kept_bottom
    | apply same_kept_ok_at, Henv3 ].
Qed.

(* Of an [Or_unknown.t]: what is known has [P]. *)
Definition known_has {A : Type} (P : A -> Prop) (a : Or_unknown.t A) : Prop :=
  match a with
  | Or_unknown.Known a => P a
  | Or_unknown.Unknown => True
  end.

Lemma meet_unknown_steady {A : Type} (P1 P2 : A -> Prop)
    (meet_contents meet_contents' :
      Meet_env.t -> A -> A -> option (Meet_result.t A))
    (contents_is_bottom : A -> bool) :
  steady P1 P2 meet_contents meet_contents' ->
  steady (known_has P1) (known_has P2)
    (meet_unknown meet_contents contents_is_bottom)
    (meet_unknown meet_contents' contents_is_bottom).
Proof.
  intros Hcontents env [a1 | ] [a2 | ] H1 H2; unfold meet_unknown;
    cbn in H1, H2.
  - destruct (contents_is_bottom a1), (contents_is_bottom a2); cbn;
      try apply same_kept_bottom.
    apply same_kept_map, Hcontents; assumption.
  - destruct (contents_is_bottom a1);
      [ apply same_kept_bottom | apply same_kept_ok ].
  - destruct (contents_is_bottom a2);
      [ apply same_kept_bottom | apply same_kept_ok ].
  - apply same_kept_ok.
Qed.

(* The same where what is known may also be bottom, for the test of
   bottom that [meet_unknown] is given: it then asks no meet. *)
Lemma meet_unknown_or_bottom_steady {A : Type} (P1 P2 : A -> Prop)
    (meet_contents meet_contents' :
      Meet_env.t -> A -> A -> option (Meet_result.t A))
    (contents_is_bottom : A -> bool) :
  steady P1 P2 meet_contents meet_contents' ->
  steady (known_has (fun a => P1 a ∨ contents_is_bottom a = true))
    (known_has (fun a => P2 a ∨ contents_is_bottom a = true))
    (meet_unknown meet_contents contents_is_bottom)
    (meet_unknown meet_contents' contents_is_bottom).
Proof.
  intros Hcontents env [a1 | ] [a2 | ] H1 H2; unfold meet_unknown;
    cbn in H1, H2.
  - destruct (contents_is_bottom a1) eqn:Hbottom1,
      (contents_is_bottom a2) eqn:Hbottom2;
      cbn; try apply same_kept_bottom.
    destruct H1 as [H1 | H1]; [ | discriminate H1 ].
    destruct H2 as [H2 | H2]; [ | discriminate H2 ].
    apply same_kept_map, Hcontents; assumption.
  - destruct (contents_is_bottom a1);
      [ apply same_kept_bottom | apply same_kept_ok ].
  - destruct (contents_is_bottom a2);
      [ apply same_kept_bottom | apply same_kept_ok ].
  - apply same_kept_ok.
Qed.

(* Of an [or_unknown_or_bottom]: what there is has [P]. *)
Definition ok_has {A : Type} (P : A -> Prop) (a : or_unknown_or_bottom A)
    : Prop :=
  match a with
  | Ok a => P a
  | Unknown | Bottom => True
  end.

Lemma meet_or_unknown_or_bottom_steady {A B : Type} (P1 P2 : A -> Prop)
    (meet_elt meet_elt' : Meet_env.t -> A -> A -> option (Meet_result.t B)) :
  steady P1 P2 meet_elt meet_elt' ->
  steady (ok_has P1) (ok_has P2) (meet_or_unknown_or_bottom meet_elt)
    (meet_or_unknown_or_bottom meet_elt').
Proof.
  intros Helt env [ | a1 | ] [ | a2 | ] H1 H2;
    unfold meet_or_unknown_or_bottom;
    try apply same_kept_bottom; try apply same_kept_ok.
  apply Helt; assumption.
Qed.

(* ======================================================================== *)
(* What an answer makes                                                     *)
(* ======================================================================== *)

(* A value that an answer makes has [Q]. Nothing is said of an answer
   that is one of the two inputs. *)
Definition answers {A : Type} (Q : A -> Prop) (o : option (Meet_result.t A))
    : Prop :=
  ∀ (a : A) (env' : Meet_env.t),
    o = Some (Meet_result.Ok (New_result a) env') -> Q a.

Lemma answers_map {A B : Type} (f : A -> B) (Q : B -> Prop)
    (o : option (Meet_result.t A)) :
  answers (fun a => Q (f a)) o -> answers Q (option_map (map_result f) o).
Proof.
  intros Ho b env' Hmap.
  destruct o as [[r | [ | | | a] env0] | ]; cbn in Hmap;
    try discriminate Hmap.
  injection Hmap as <- _. exact (Ho a env0 eq_refl).
Qed.

(* The same for an answer that is not optional, where every value that
   is made has [Q]. *)
Lemma answers_map_some {A B : Type} (f : A -> B) (Q : B -> Prop)
    (result : Meet_result.t A) :
  (∀ a : A, Q (f a)) -> answers Q (Some (map_result f result)).
Proof.
  intros HQ b env' Hmap.
  destruct result as [r | [ | | | a] env0]; cbn in Hmap;
    try discriminate Hmap.
  injection Hmap as <- _. apply HQ.
Qed.

(* What an answer is made of has [Q], where the inputs have. *)
Lemma extract_has {A : Type} (Q : A -> Prop) (r : meet_return_value A)
    (left right : A) (env : Meet_env.t) (o : option (Meet_result.t A)) :
  Q left -> Q right -> answers Q o -> o = Some (Meet_result.Ok r env) ->
  Q (extract_value r left right).
Proof.
  intros Hleft Hright Ho ->.
  destruct r as [ | | | a]; cbn; try assumption. exact (Ho a env eq_refl).
Qed.

(* A value that a pair of answers makes is the one computed. *)
Lemma combine_new {A B C : Type} (a : meet_return_value A)
    (b : meet_return_value B) (compute_value : unit -> C) (c : C) :
  combine_meet_return_values a b compute_value = New_result c ->
  c = compute_value tt.
Proof.
  destruct a, b; cbn; intros Hc; try discriminate Hc; injection Hc as <-;
    reflexivity.
Qed.

Lemma combine_results2_answers {A B C : Type} (QA : A -> Prop)
    (QB : B -> Prop) (Q : C -> Prop)
    (meet_a : Meet_env.t -> A -> A -> option (Meet_result.t A))
    (meet_b : Meet_env.t -> B -> B -> option (Meet_result.t B))
    (env : Meet_env.t) (left_a right_a : A) (left_b right_b : B)
    (rebuild : A -> B -> C) :
  QA left_a -> QA right_a -> QB left_b -> QB right_b ->
  answers QA (meet_a env left_a right_a) ->
  (∀ env1 : Meet_env.t, answers QB (meet_b env1 left_b right_b)) ->
  (∀ (a : A) (b : B), QA a -> QB b -> Q (rebuild a b)) ->
  answers Q
    (combine_results2 env meet_a meet_b left_a right_a left_b right_b rebuild).
Proof.
  intros Hla Hra Hlb Hrb Ha Hb Hrebuild c env' Hc.
  unfold combine_results2 in Hc.
  destruct (meet_a env left_a right_a) as [[ra | ra env1] | ] eqn:Hmeet_a;
    cbv beta iota in Hc; try discriminate Hc.
  destruct (meet_b env1 left_b right_b) as [[rb | rb env2] | ] eqn:Hmeet_b;
    cbv beta iota in Hc; try discriminate Hc.
  injection Hc as Hc _. apply combine_new in Hc. subst c.
  apply Hrebuild.
  - exact (extract_has QA ra left_a right_a env1 _ Hla Hra Ha eq_refl).
  - exact (extract_has QB rb left_b right_b env2 _ Hlb Hrb (Hb env1) Hmeet_b).
Qed.

Lemma meet_unknown_answers {A : Type} (Q : A -> Prop)
    (meet_contents : Meet_env.t -> A -> A -> option (Meet_result.t A))
    (contents_is_bottom : A -> bool) (env : Meet_env.t)
    (a1 a2 : Or_unknown.t A) :
  (∀ c1 c2 : A,
    a1 = Or_unknown.Known c1 -> a2 = Or_unknown.Known c2 ->
    answers Q (meet_contents env c1 c2)) ->
  answers (known_has Q)
    (meet_unknown meet_contents contents_is_bottom env a1 a2).
Proof.
  intros Hcontents a env' Ha. unfold meet_unknown in Ha.
  destruct a1 as [c1 | ], a2 as [c2 | ].
  - destruct (contents_is_bottom c1), (contents_is_bottom c2); cbn in Ha;
      try discriminate Ha.
    exact
      (answers_map Or_unknown.Known (known_has Q) _
         (Hcontents c1 c2 eq_refl eq_refl) a env' Ha).
  - destruct (contents_is_bottom c1); discriminate Ha.
  - destruct (contents_is_bottom c2); discriminate Ha.
  - discriminate Ha.
Qed.

Lemma meet_or_unknown_or_bottom_answers {A B : Type} (Q : B -> Prop)
    (meet_elt : Meet_env.t -> A -> A -> option (Meet_result.t B))
    (env : Meet_env.t) (input1 input2 : or_unknown_or_bottom A) :
  (∀ elt1 elt2 : A,
    input1 = Ok elt1 -> input2 = Ok elt2 ->
    answers Q (meet_elt env elt1 elt2)) ->
  answers Q (meet_or_unknown_or_bottom meet_elt env input1 input2).
Proof.
  intros Helt b env' Hb.
  destruct input1 as [ | elt1 | ], input2 as [ | elt2 | ]; cbn in Hb;
    try discriminate Hb.
  exact (Helt elt1 elt2 eq_refl eq_refl b env' Hb).
Qed.

(* What [meet_mapping] asks of the fold that it is given, on inputs
   that have [PM]: it is a fold over a list of keys, each with what the
   two inputs have for it, and every component that it finds has
   [P]. *)
Definition opt_has {A : Type} (P : A -> Prop) (a : option A) : Prop :=
  match a with
  | Some a => P a
  | None => True
  end.

Definition fold2_over {K A M : Type} (PM1 PM2 : M -> Prop)
    (P1 P2 : A -> Prop)
    (fold2 :
      (K -> option A -> option A -> mapping_state K A -> mapping_state K A) ->
      M -> M -> mapping_state K A -> mapping_state K A) : Prop :=
  ∀ left right : M,
    PM1 left -> PM2 right ->
    ∃ entries : list (K * (option A * option A)),
      (∀ (step :
            K -> option A -> option A -> mapping_state K A ->
            mapping_state K A)
          (init : mapping_state K A),
          fold2 step left right init
          = List.fold_left
              (fun state entry =>
                 step (fst entry) (fst (snd entry)) (snd (snd entry)) state)
              entries init)
      ∧ ∀ entry : K * (option A * option A),
          In entry entries ->
          opt_has P1 (fst (snd entry)) ∧ opt_has P2 (snd (snd entry)).

(* The state of [meet_mapping] has the typing environment of [env], if
   it has an environment. *)
Definition state_kept {K A : Type} (env : Meet_env.t)
    (state : mapping_state K A) : Prop :=
  ∀ acc : mapping_acc K A,
    state = Some (Or_bottom.Ok acc) ->
    Meet_env.typing_env (mapping_env acc) = Meet_env.typing_env env.

(* The step of [meet_mapping]. *)
Definition mapping_step {K A : Type}
    (meet_data : Meet_env.t -> A -> A -> option (Meet_result.t A)) (key : K)
    (data_left_opt data_right_opt : option A) (state : mapping_state K A)
    : mapping_state K A :=
  match state with
  | None => None
  | Some Or_bottom.Bottom => Some Or_bottom.Bottom
  | Some (Or_bottom.Ok acc) =>
    match data_left_opt, data_right_opt with
    | None, None => None
    | Some data_left, None =>
      Some
        (Or_bottom.Ok
           {| all_left := all_left acc;
              all_right := false;
              mapping := (key, data_left) :: mapping acc;
              mapping_env := mapping_env acc |})
    | None, Some data_right =>
      Some
        (Or_bottom.Ok
           {| all_left := false;
              all_right := all_right acc;
              mapping := (key, data_right) :: mapping acc;
              mapping_env := mapping_env acc |})
    | Some data_left, Some data_right =>
      match meet_data (mapping_env acc) data_left data_right with
      | None => None
      | Some (Meet_result.Bottom _) => Some Or_bottom.Bottom
      | Some (Meet_result.Ok res env) =>
        let result (all_left all_right : bool) (data : A)
            : mapping_state K A :=
          Some
            (Or_bottom.Ok
               {| all_left := all_left;
                  all_right := all_right;
                  mapping := (key, data) :: mapping acc;
                  mapping_env := env |})
        in
        match res with
        | Both_inputs => result (all_left acc) (all_right acc) data_left
        | Left_input => result (all_left acc) false data_left
        | Right_input => result false (all_right acc) data_right
        | New_result data => result false false data
        end
      end
    end
  end.

Lemma meet_mapping_unfold {K A M : Type}
    (meet_data : Meet_env.t -> A -> A -> option (Meet_result.t A))
    (fold2 :
      (K -> option A -> option A -> mapping_state K A -> mapping_state K A) ->
      M -> M -> mapping_state K A -> mapping_state K A)
    (env : Meet_env.t) (left right : M) (rebuild : list (K * A) -> M) :
  meet_mapping meet_data fold2 env left right rebuild
  = match
      fold2 (mapping_step meet_data) left right
        (Some
           (Or_bottom.Ok
              {| all_left := true;
                 all_right := true;
                 mapping := [];
                 mapping_env := env |}))
    with
    | None => None
    | Some Or_bottom.Bottom => Some (Meet_result.Bottom (New_result tt))
    | Some (Or_bottom.Ok res) =>
      let result :=
        match all_left res, all_right res with
        | true, true => Both_inputs
        | true, false => Left_input
        | false, true => Right_input
        | false, false => New_result (rebuild (mapping res))
        end
      in
      Some (Meet_result.Ok result (mapping_env res))
    end.
Proof. reflexivity. Qed.

Lemma mapping_step_steady {K A : Type} (P1 P2 : A -> Prop)
    (meet_data meet_data' : Meet_env.t -> A -> A -> option (Meet_result.t A))
    (env : Meet_env.t) (key : K) (a b : option A)
    (state : mapping_state K A) :
  steady P1 P2 meet_data meet_data' ->
  opt_has P1 a -> opt_has P2 b ->
  state_kept env state ->
  mapping_step meet_data key a b state = mapping_step meet_data' key a b state
  ∧ state_kept env (mapping_step meet_data key a b state).
Proof.
  intros Hdata Ha Hb Hstate. unfold mapping_step.
  destruct state as [[acc | ] | ];
    [ | split; [ reflexivity | intros acc'; discriminate ]
      | split; [ reflexivity | intros acc'; discriminate ] ].
  pose proof (Hstate acc eq_refl) as Henv.
  destruct a as [a | ], b as [b | ].
  - cbn in Ha, Hb.
    pose proof (Hdata (mapping_env acc) a b Ha Hb) as Hmeet.
    kept_cases Hmeet env1 Henv1.
    + split; [ reflexivity | intros acc'; discriminate ].
    + split; [ reflexivity | intros acc'; discriminate ].
    + cbv zeta. rewrite Henv in Henv1.
      destruct r; (split; [ reflexivity | intros acc' [= <-]; exact Henv1 ]).
  - split; [ reflexivity | intros acc' [= <-]; exact Henv ].
  - split; [ reflexivity | intros acc' [= <-]; exact Henv ].
  - split; [ reflexivity | intros acc'; discriminate ].
Qed.

Lemma meet_mapping_steady {K A M : Type} (PM1 PM2 : M -> Prop)
    (P1 P2 : A -> Prop)
    (meet_data meet_data' : Meet_env.t -> A -> A -> option (Meet_result.t A))
    (fold2 :
      (K -> option A -> option A -> mapping_state K A -> mapping_state K A) ->
      M -> M -> mapping_state K A -> mapping_state K A)
    (rebuild : list (K * A) -> M) :
  steady P1 P2 meet_data meet_data' ->
  fold2_over PM1 PM2 P1 P2 fold2 ->
  steady PM1 PM2
    (fun env left right => meet_mapping meet_data fold2 env left right rebuild)
    (fun env left right =>
       meet_mapping meet_data' fold2 env left right rebuild).
Proof.
  intros Hdata Hfold env left right Hleft Hright.
  destruct (Hfold left right Hleft Hright) as (entries & Hfold_eq & Hentries).
  rewrite !meet_mapping_unfold, !Hfold_eq. clear Hfold_eq.
  match goal with
  | |- same_kept _ match List.fold_left _ _ ?init with _ => _ end _ =>
    assert (Hinit : state_kept env init)
      by (intros acc [= <-]; reflexivity);
    generalize dependent init
  end.
  induction entries as [ | [key [a b]] entries IH]; intros state Hstate.
  - cbn [List.fold_left].
    destruct state as [[res | ] | ];
      [ | apply same_kept_bottom | apply same_kept_none ].
    cbv zeta. apply same_kept_ok_at, (Hstate res eq_refl).
  - destruct (Hentries (key, (a, b)) (or_introl eq_refl)) as [Ha Hb].
    cbn [List.fold_left fst snd] in *.
    destruct (mapping_step_steady P1 P2 meet_data meet_data' env key a b state
                Hdata Ha Hb Hstate) as [<- Hstate'].
    apply IH; [ | exact Hstate' ].
    intros entry Hin. apply Hentries. right. exact Hin.
Qed.

Lemma keyed_lookup_has {K A : Type} (eqb : K -> K -> bool) (P : A -> Prop)
    (m : Keyed.finite_map K A) (key : K) :
  (∀ binding : K * A, In binding m -> P (snd binding)) ->
  opt_has P (Keyed.lookup eqb key m).
Proof.
  intros Hall. induction m as [ | [k a] m IH]; cbn; [ exact I | ].
  destruct (eqb key k).
  - exact (Hall (k, a) (or_introl eq_refl)).
  - apply IH. intros binding Hin. apply Hall. right. exact Hin.
Qed.

Lemma fold_left_map_entries {S K E : Type} (g : S -> E -> S) (h : K -> E)
    (keys : list K) (init : S) :
  List.fold_left (fun s k => g s (h k)) keys init
  = List.fold_left g (map h keys) init.
Proof.
  revert init. induction keys as [ | k keys IH]; intros init; simpl;
    [ reflexivity | apply IH ].
Qed.

(* Every component of a map has [P]. *)
Definition bindings_have {K A : Type} (P : A -> Prop)
    (m : Keyed.finite_map K A) : Prop :=
  ∀ binding : K * A, In binding m -> P (snd binding).

Lemma map_meet_steady {K A : Type} (eqb : K -> K -> bool)
    (P1 P2 : A -> Prop)
    (meet_data meet_data' : Meet_env.t -> A -> A -> option (Meet_result.t A)) :
  steady P1 P2 meet_data meet_data' ->
  steady (bindings_have P1) (bindings_have P2) (map_meet eqb meet_data)
    (map_meet eqb meet_data').
Proof.
  intros Hdata. unfold map_meet. cbv zeta.
  apply (meet_mapping_steady (bindings_have P1) (bindings_have P2) P1 P2);
    [ exact Hdata | ].
  intros left right Hleft Hright.
  exists
    (map
       (fun key =>
          (key, (Keyed.lookup eqb key left, Keyed.lookup eqb key right)))
       (Keyed.keys_of_both eqb left right)).
  split.
  - intros step init. unfold Keyed.fold2_on.
    rewrite <- fold_left_map_entries. reflexivity.
  - intros entry Hin. apply in_map_iff in Hin. destruct Hin as (key & <- & _).
    cbn. split; apply keyed_lookup_has; assumption.
Qed.

(* The type of an expanded type is not an alias. *)
Lemma to_type_not_alias (e : Expanded_type.t) :
  get_alias_opt (Expanded_type.to_type e) = None.
Proof.
  destruct e as [kind descr]. unfold Expanded_type.to_type.
  cbn [Expanded_type.descr Expanded_type.kind].
  destruct descr as [ | head | ]; [ | destruct head; reflexivity | ];
    destruct kind as [ | [] | | ]; reflexivity.
Qed.

Lemma nameless_expanded_succ (depth : nat) (e : Expanded_type.t) :
  nameless_expanded_within (S depth) e
  = nameless_type (nameless_within depth) (Expanded_type.to_type e).
Proof.
  unfold nameless_expanded_within. cbn [nameless_within].
  unfold nameless_or_const. rewrite to_type_not_alias. reflexivity.
Qed.

(* A type that has no name inside is not an alias, and its head has
   none; or it is an alias of a constant, and the head that the
   constant expands to has none. *)
Lemma nameless_cases (depth : nat) (ty : flambda_type) :
  nameless_within depth ty = true ->
  (∃ e : Expanded_type.t,
    Expanded_type.of_non_alias_type ty = Some e
    ∧ nameless_expanded_within depth e = true)
  ∨ ∃ c : const,
      get_alias_opt ty = Some (Const c)
      ∧ nameless_expanded_within depth (Expanded_type.create_const c) = true.
Proof.
  destruct depth as [ | d]; [ discriminate | ]. cbn [nameless_within].
  unfold nameless_or_const.
  destruct (get_alias_opt ty) as [[v | s | c] | ] eqn:Halias;
    try discriminate.
  - intros Hc. right. exists c. split; [ reflexivity | ].
    rewrite nameless_expanded_succ. exact Hc.
  - intros Hty. left.
    destruct ty as [ty | ty | k ty | ty | ty];
      destruct ty as [ | [head | s] | ]; cbn in Halias;
      try discriminate Halias; eexists; (split; [ reflexivity | ]);
      rewrite nameless_expanded_succ; cbn; try exact Hty; try reflexivity.
    all: destruct k; reflexivity.
Qed.

(* The canonical element of a constant is the constant. *)
Lemma canonical_of_const_simple (tenv : Typing_env.t) (c : const) :
  Typing_env.get_canonical_simple_exn tenv (Some Name_mode.in_types) None
    (Const c)
  = Some (Some (Const c)).
Proof. reflexivity. Qed.

(* Two constants are not made aliases: a constant is not demoted. *)
Lemma aliases_add_constants {A : Type}
    (resolver : name -> Binding_time.With_name_mode.t)
    (types : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) (c1 c2 : const) :
  Aliases.add resolver types t (Const c1) (Const c2) = None
  ∨ Aliases.add resolver types t (Const c1) (Const c2)
    = Some Or_bottom.Bottom.
Proof.
  unfold Aliases.add.
  destruct (Aliases.add0 _ _ _ _ _) as [[[t' which] | ] | ]; cbn;
    [ | right; reflexivity | left; reflexivity ].
  destruct which; cbn; left; reflexivity.
Qed.

Lemma add_alias_constants (tenv : Typing_env.t) (c1 c2 : const) :
  Typing_env.add_alias tenv (Const c1) (Const c2) = None
  ∨ Typing_env.add_alias tenv (Const c1) (Const c2) = Some Bottom.
Proof.
  unfold Typing_env.add_alias, Typing_env.aliases_add.
  destruct
    (aliases_add_constants (Typing_env.binding_time_resolver tenv)
       (Typing_env.names_to_types tenv) (Typing_env.aliases tenv) c1 c2)
    as [-> | ->]; [ left | right ]; reflexivity.
Qed.

(* The meet of two aliases of constants asks no meet. *)
Lemma meet_type_constants (menv : Meet_env.t) (t1 t2 : flambda_type)
    (c1 c2 : const) (mh : meet_expanded_head) :
  get_alias_opt t1 = Some (Const c1) ->
  get_alias_opt t2 = Some (Const c2) ->
  Meet_env.meet_type menv t1 t2 mh
  = if negb (kind_eqb (kind t1) (kind t2))
    then None
    else if simple_equal (Const c1) (Const c2)
    then Some (Meet_result.Ok Both_inputs menv)
    else
      match
        Typing_env.add_alias (Meet_env.typing_env menv) (Const c1) (Const c2)
      with
      | Some Bottom => Some (Meet_result.Bottom (New_result tt))
      | _ => None
      end.
Proof.
  intros H1 H2.
  unfold Meet_env.meet_type, Meet_env.meet,
    Typing_env.get_alias_then_canonical_simple_exn.
  destruct (negb (kind_eqb (kind t1) (kind t2))); [ reflexivity | ].
  rewrite H1, H2, !canonical_of_const_simple. cbn beta iota.
  unfold Meet_env.add_alias_between_canonicals.
  destruct (simple_equal (Const c1) (Const c2)); [ reflexivity | ].
  destruct (add_alias_constants (Meet_env.typing_env menv) c1 c2)
    as [-> | ->]; reflexivity.
Qed.

(* The meet of a type that is not an alias with an alias of a constant
   asks one meet: that of the head of the type with the head that the
   constant expands to. *)
Lemma meet_type_constant_right (menv : Meet_env.t) (t1 t2 : flambda_type)
    (e1 : Expanded_type.t) (c2 : const) (mh : meet_expanded_head) :
  Expanded_type.of_non_alias_type t1 = Some e1 ->
  get_alias_opt t2 = Some (Const c2) ->
  Meet_env.meet_type menv t1 t2 mh
  = if negb (kind_eqb (kind t1) (kind t2))
    then None
    else
      match mh menv e1 (Expanded_type.create_const c2) with
      | None => None
      | Some (Meet_result.Ok _ env) => Some (Meet_result.Ok Right_input env)
      | Some (Meet_result.Bottom _) =>
        Some (Meet_result.Bottom (New_result tt))
      end.
Proof.
  intros H1 H2.
  unfold Meet_env.meet_type, Meet_env.meet,
    Typing_env.get_alias_then_canonical_simple_exn, expand_head0.
  destruct (negb (kind_eqb (kind t1) (kind t2))); [ reflexivity | ].
  rewrite (of_non_alias_type_not_alias t1 e1 H1), H1, H2,
    canonical_of_const_simple.
  cbn. unfold Meet_env.add_concrete_equation_on_const.
  destruct (mh menv e1 (Expanded_type.create_const c2)) as [[r | r env] | ];
    reflexivity.
Qed.

Lemma meet_type_constant_left (menv : Meet_env.t) (t1 t2 : flambda_type)
    (c1 : const) (e2 : Expanded_type.t) (mh : meet_expanded_head) :
  get_alias_opt t1 = Some (Const c1) ->
  Expanded_type.of_non_alias_type t2 = Some e2 ->
  Meet_env.meet_type menv t1 t2 mh
  = if negb (kind_eqb (kind t1) (kind t2))
    then None
    else
      match mh menv e2 (Expanded_type.create_const c1) with
      | None => None
      | Some (Meet_result.Ok _ env) => Some (Meet_result.Ok Left_input env)
      | Some (Meet_result.Bottom _) =>
        Some (Meet_result.Bottom (New_result tt))
      end.
Proof.
  intros H1 H2.
  unfold Meet_env.meet_type, Meet_env.meet,
    Typing_env.get_alias_then_canonical_simple_exn, expand_head0.
  destruct (negb (kind_eqb (kind t1) (kind t2))); [ reflexivity | ].
  rewrite (of_non_alias_type_not_alias t2 e2 H2), H1, H2,
    canonical_of_const_simple.
  cbn. unfold Meet_env.add_concrete_equation_on_const.
  destruct (mh menv e2 (Expanded_type.create_const c1)) as [[r | r env] | ];
    reflexivity.
Qed.

(* ======================================================================== *)
(* The step of the meet                                                     *)
(* ======================================================================== *)

(* What is asked of a type inside, as a proposition. *)
Definition has (inside : flambda_type -> bool) (ty : flambda_type) : Prop :=
  inside ty = true.

(* The same of every type of a list. *)
Definition all_have (inside : flambda_type -> bool)
    (fields : list flambda_type) : Prop :=
  ∀ ty : flambda_type, In ty fields -> has inside ty.

Lemma forallb_bindings_have {K : Type} (inside : flambda_type -> bool)
    (m : Keyed.finite_map K flambda_type) :
  forallb (fun binding => inside (snd binding)) m = true ->
  bindings_have (has inside) m.
Proof.
  intros Hall binding Hin. rewrite forallb_forall in Hall.
  exact (Hall binding Hin).
Qed.

Lemma meet_function_type_steady :
  steady (fun _ : Or_unknown.t function_type => True) (fun _ => True)
    meet_function_type meet_function_type.
Proof.
  intros env [f1 | ] [f2 | ] _ _; unfold meet_function_type;
    try apply same_kept_ok.
  unfold meet_code_id. destruct (Pos.eqb _ _); cbn;
    [ apply same_kept_ok | apply same_kept_bottom ].
Qed.

(* The step is taken for two meets of less fuel, [mh] and [mh'], and
   two classes of heads, one for each side: on the left a head may
   have variables for relations if [relations1] says so and its types
   inside have [inside1], and the same on the right. What is supposed:
   not both sides may have variables for relations, and the meets of
   types that are made of [mh] and [mh'] answer the same, with the
   typing environment given, on a type that has [inside1] and one
   that has [inside2]. *)
Section Step.

Variable join : join_scoped_envs.
Variable choices : meet_choices.
Variable mh mh' : meet_expanded_head.
Variables relations1 relations2 : bool.
Variables inside1 inside2 : flambda_type -> bool.
Hypothesis Hjoin : join_of_nothing join.
Hypothesis Hrelations : relations1 && relations2 = false.
Hypothesis Hinside :
  steady (has inside1) (has inside2) (FlambdaMeet.meet mh)
    (FlambdaMeet.meet mh').

Lemma meet_array_of_types_steady (length : nat) :
  steady (all_have inside1) (all_have inside2)
    (fun env fields1 fields2 =>
       meet_array_of_types mh env fields1 fields2 length)
    (fun env fields1 fields2 =>
       meet_array_of_types mh' env fields1 fields2 length).
Proof.
  unfold meet_array_of_types. cbv zeta.
  apply
    (meet_mapping_steady (all_have inside1) (all_have inside2) (has inside1)
       (has inside2));
    [ exact Hinside | ].
  intros left right Hleft Hright.
  exists
    (map (fun i => (i, (nth_error left i, nth_error right i)))
       (seq 0 length)).
  split.
  - intros step init. rewrite <- fold_left_map_entries. reflexivity.
  - intros entry Hin. apply in_map_iff in Hin. destruct Hin as (i & <- & _).
    cbn.
    split;
      [ destruct (nth_error left i) eqn:Hnth
      | destruct (nth_error right i) eqn:Hnth ];
      cbn; try exact I; [ apply Hleft | apply Hright ];
      exact (nth_error_In _ _ Hnth).
Qed.

Lemma meet_int_indexed_product_steady :
  steady (all_have inside1) (all_have inside2) (meet_int_indexed_product mh)
    (meet_int_indexed_product mh').
Proof.
  intros env fields1 fields2 H1 H2. unfold meet_int_indexed_product.
  exact (meet_array_of_types_steady _ env fields1 fields2 H1 H2).
Qed.

Lemma meet_product_function_slot_indexed_steady :
  steady (bindings_have (has inside1)) (bindings_have (has inside2))
    (meet_product_function_slot_indexed mh)
    (meet_product_function_slot_indexed mh').
Proof.
  intros env c1 c2 H1 H2. unfold meet_product_function_slot_indexed.
  exact
    (map_meet_steady Pos.eqb (has inside1) (has inside2) _ _ Hinside env c1 c2
       H1 H2).
Qed.

Lemma meet_product_value_slot_indexed_steady :
  steady (bindings_have (has inside1)) (bindings_have (has inside2))
    (meet_product_value_slot_indexed mh)
    (meet_product_value_slot_indexed mh').
Proof.
  intros env c1 c2 H1 H2. unfold meet_product_value_slot_indexed.
  exact
    (map_meet_steady Pos.eqb (has inside1) (has inside2) _ _ Hinside env c1 c2
       H1 H2).
Qed.

Lemma meet_closures_entry_steady :
  steady (fun entry => plain_closures_entry inside1 entry = true)
    (fun entry => plain_closures_entry inside2 entry = true)
    (meet_closures_entry mh) (meet_closures_entry mh').
Proof.
  intros env [ft1 ct1 vt1] [ft2 ct2 vt2] H1 H2. unfold meet_closures_entry.
  cbn in H1, H2. apply andb_true_iff in H1, H2.
  destruct H1 as [Hc1 Hv1], H2 as [Hc2 Hv2].
  apply
    (combine_results3_steady (bindings_have (fun _ => True))
       (bindings_have (fun _ => True)) (bindings_have (has inside1))
       (bindings_have (has inside2)) (bindings_have (has inside1))
       (bindings_have (has inside2)));
    try (apply forallb_bindings_have; assumption);
    try (intros binding _; exact I).
  - apply map_meet_steady, meet_function_type_steady.
  - exact meet_product_function_slot_indexed_steady.
  - exact meet_product_value_slot_indexed_steady.
Qed.

(* What is cut from an environment that was entered and to which
   nothing was added. *)
Lemma cut_entered (tenv : Typing_env.t) :
  Typing_env.cut (Typing_env.increment_scope tenv)
    (Typing_env.current_scope tenv)
  = Some Typing_env_level.empty.
Proof.
  unfold Typing_env.cut.
  change (Typing_env.current_scope (Typing_env.increment_scope tenv))
    with (Scope.next (Typing_env.current_scope tenv)).
  unfold Scope.next.
  destruct
    (Z.leb_spec (Typing_env.current_scope tenv + 1)
       (Typing_env.current_scope tenv)) as [Hle | _]; [ lia | ].
  cbn [Typing_env.increment_scope Typing_env.prev_levels
       Typing_env.current_level Typing_env.One_level.level
       Typing_env.cut_loop].
  change (Typing_env.One_level.scope (Typing_env.current_level tenv))
    with (Typing_env.current_scope tenv).
  rewrite Z.ltb_irrefl. reflexivity.
Qed.

Lemma add_empty_extension_strict (env : Meet_env.t)
    (mh0 : meet_expanded_head) :
  Meet_env.add_env_extension_strict env (Env_extension []) mh0
  = Some
      (if Typing_env.is_bottom (Meet_env.typing_env env)
       then Or_bottom.Bottom
       else Or_bottom.Ok env).
Proof.
  unfold Meet_env.add_env_extension_strict.
  destruct (Typing_env.is_bottom _); reflexivity.
Qed.

Lemma extension_of_empty_level :
  extension_of_level Typing_env_level.empty = Env_extension [].
Proof. reflexivity. Qed.

Lemma add_defined_vars_empty (env : Meet_env.t) :
  add_defined_vars choices env Typing_env_level.empty = Some env.
Proof. reflexivity. Qed.

Lemma meet_disjunction_steady {A B : Type} (PA1 PA2 : A -> Prop)
    (PB1 PB2 : B -> Prop)
    (meet_a meet_a' : Meet_env.t -> A -> A -> option (Meet_result.t A))
    (meet_b meet_b' : Meet_env.t -> B -> B -> option (Meet_result.t B))
    (bottom_a : unit -> A) (bottom_b : unit -> B) (env : Meet_env.t)
    (a1 : A) (b1 : B) (a2 : A) (b2 : B) :
  steady PA1 PA2 meet_a meet_a' ->
  steady PB1 PB2 meet_b meet_b' ->
  PA1 a1 -> PA2 a2 -> PB1 b1 -> PB2 b2 ->
  same_kept env
    (meet_disjunction join choices mh meet_a meet_b bottom_a bottom_b env a1
       b1 No_extensions a2 b2 No_extensions)
    (meet_disjunction join choices mh' meet_a' meet_b' bottom_a bottom_b env
       a1 b1 No_extensions a2 b2 No_extensions).
Proof.
  intros Ha Hb Ha1 Ha2 Hb1 Hb2. unfold meet_disjunction.
  unfold Meet_env.enter_scope. cbv zeta. cbv beta iota.
  set (scoped :=
    {| Meet_env.typing_env :=
         Typing_env.increment_scope (Meet_env.typing_env env);
       Meet_env.adding_equations_for_names :=
         Meet_env.adding_equations_for_names env;
       Meet_env.names_to_avoid := Meet_env.names_to_avoid env |}).
  (* An environment that has the typing environment that was entered:
     nothing is cut from it. *)
  assert (Hcut :
    ∀ env1 : Meet_env.t,
      Meet_env.typing_env env1 = Meet_env.typing_env scoped ->
      Typing_env.cut (Meet_env.typing_env env1)
        (Typing_env.current_scope (Meet_env.typing_env env))
      = Some Typing_env_level.empty).
  { intros env1 Henv1. rewrite Henv1. apply cut_entered. }
  (* What is answered where one alternative is bottom. *)
  assert (Hdirect :
    ∀ (C : Type) (result : meet_return_value C) (env1 : Meet_env.t),
      Meet_env.typing_env env1 = Meet_env.typing_env scoped ->
      same_kept env
        (let* level :=
           Typing_env.cut (Meet_env.typing_env env1)
             (Typing_env.current_scope (Meet_env.typing_env env))
         in
         let* initial_env :=
           add_defined_vars choices
             (Meet_env.with_names_to_avoid env (Meet_env.names_to_avoid env1))
             level
         in
         match
           Meet_env.add_env_extension_strict initial_env
             (extension_of_level level) mh
         with
         | Some (Or_bottom.Ok env2) => Some (Meet_result.Ok result env2)
         | Some Or_bottom.Bottom =>
           Some (Meet_result.Bottom (map_return_value (fun _ => tt) result))
         | None => None
         end)
        (let* level :=
           Typing_env.cut (Meet_env.typing_env env1)
             (Typing_env.current_scope (Meet_env.typing_env env))
         in
         let* initial_env :=
           add_defined_vars choices
             (Meet_env.with_names_to_avoid env (Meet_env.names_to_avoid env1))
             level
         in
         match
           Meet_env.add_env_extension_strict initial_env
             (extension_of_level level) mh'
         with
         | Some (Or_bottom.Ok env2) => Some (Meet_result.Ok result env2)
         | Some Or_bottom.Bottom =>
           Some (Meet_result.Bottom (map_return_value (fun _ => tt) result))
         | None => None
         end)).
  { intros C result env1 Henv1.
    rewrite (Hcut env1 Henv1), add_defined_vars_empty,
      extension_of_empty_level, !add_empty_extension_strict.
    destruct (Typing_env.is_bottom _);
      [ apply same_kept_bottom | apply same_kept_ok_at; reflexivity ]. }
  pose proof (Ha scoped a1 a2 Ha1 Ha2) as Hfirst.
  kept_cases Hfirst env_a Henv_a.
  - apply same_kept_none.
  - cbv beta iota.
    pose proof (Hb scoped b1 b2 Hb1 Hb2) as Hsecond.
    kept_cases Hsecond env_b Henv_b.
    + apply same_kept_none.
    + apply same_kept_bottom.
    + unfold Meet_env.final_typing_env_strict. cbv beta iota.
      apply Hdirect, Henv_b.
  - unfold Meet_env.final_typing_env_strict. cbv beta iota.
    set (scoped_b :=
      Meet_env.with_names_to_avoid scoped (Meet_env.names_to_avoid env_a)).
    pose proof (Hb scoped_b b1 b2 Hb1 Hb2) as Hsecond.
    kept_cases Hsecond env_b Henv_b.
    + apply same_kept_none.
    + apply Hdirect, Henv_a.
    + cbv beta iota.
      destruct
        (Hjoin
           (Meet_env.with_names_to_avoid env (Meet_env.names_to_avoid env_b))
           [Meet_env.final_typing_env env_a; Meet_env.final_typing_env env_b])
        as (result & Hresult & Hjoined);
        [ discriminate
        | intros scoped_env [<- | [<- | []]];
            [ exact Henv_a | exact Henv_b ]
        | ].
      assert (Hjoined' :
        ∀ mh0 : meet_expanded_head,
          join mh0 (Typing_env.current_scope (Meet_env.typing_env env))
            (Meet_env.with_names_to_avoid env (Meet_env.names_to_avoid env_b))
            (Meet_env.typing_env env)
            [Meet_env.final_typing_env env_a; Meet_env.final_typing_env env_b]
          = Some result) by exact Hjoined.
      rewrite !Hjoined'. unfold Meet_env.final_typing_env.
      rewrite (Hcut env_a Henv_a), (Hcut env_b Henv_b).
      cbv beta iota. rewrite !add_defined_vars_empty. cbv beta iota.
      change (Typing_env_level.is_empty Typing_env_level.empty) with true.
      cbn [andb]. apply same_kept_ok_at. exact Hresult.
Qed.

(* What the meet of a disjunction makes, where neither input has an
   extension: the two alternatives as they are answered, or the bottom
   of one, and no extension, as nothing is added to the environment of
   an alternative. *)
Lemma meet_disjunction_answers {A B : Type} (QA : A -> Prop)
    (QB : B -> Prop)
    (meet_a : Meet_env.t -> A -> A -> option (Meet_result.t A))
    (meet_b : Meet_env.t -> B -> B -> option (Meet_result.t B))
    (bottom_a : unit -> A) (bottom_b : unit -> B) (env : Meet_env.t)
    (a1 : A) (b1 : B) (a2 : A) (b2 : B) :
  (∀ env1 : Meet_env.t, kept env1 (meet_a env1 a1 a2)) ->
  (∀ env1 : Meet_env.t, kept env1 (meet_b env1 b1 b2)) ->
  QA a1 -> QA a2 -> QB b1 -> QB b2 -> QA (bottom_a tt) -> QB (bottom_b tt) ->
  (∀ env1 : Meet_env.t, answers QA (meet_a env1 a1 a2)) ->
  (∀ env1 : Meet_env.t, answers QB (meet_b env1 b1 b2)) ->
  answers
    (fun result : A * B * variant_extensions =>
       QA (fst (fst result)) ∧ QB (snd (fst result))
       ∧ snd result = No_extensions)
    (meet_disjunction join choices mh meet_a meet_b bottom_a bottom_b env a1
       b1 No_extensions a2 b2 No_extensions).
Proof.
  intros Ha Hb HQa1 HQa2 HQb1 HQb2 HQa0 HQb0 Hanswers_a Hanswers_b made env'
    Hmade.
  unfold meet_disjunction in Hmade.
  unfold Meet_env.enter_scope, Meet_env.final_typing_env_strict in Hmade.
  cbv zeta in Hmade. cbv beta iota in Hmade.
  set (scoped :=
    {| Meet_env.typing_env :=
         Typing_env.increment_scope (Meet_env.typing_env env);
       Meet_env.adding_equations_for_names :=
         Meet_env.adding_equations_for_names env;
       Meet_env.names_to_avoid := Meet_env.names_to_avoid env |}) in Hmade.
  assert (Hcut :
    ∀ env1 : Meet_env.t,
      Meet_env.typing_env env1 = Meet_env.typing_env scoped ->
      Typing_env.cut (Meet_env.typing_env env1)
        (Typing_env.current_scope (Meet_env.typing_env env))
      = Some Typing_env_level.empty).
  { intros env1 Henv1. rewrite Henv1. apply cut_entered. }
  pose proof (Ha scoped) as Hkept_a.
  destruct (meet_a scoped a1 a2) as [[ra | ra env_a] | ] eqn:Hmeet_a;
    cbv beta iota in Hmade; [ | | discriminate Hmade ].
  - (* The first alternative is bottom. *)
    pose proof (Hb scoped) as Hkept_b.
    destruct (meet_b scoped b1 b2) as [[rb | rb env_b] | ] eqn:Hmeet_b;
      cbv beta iota in Hmade; try discriminate Hmade.
    rewrite (Hcut env_b (Hkept_b rb env_b eq_refl)) in Hmade.
    cbv beta iota in Hmade.
    rewrite add_defined_vars_empty, extension_of_empty_level,
      add_empty_extension_strict in Hmade.
    cbv beta iota in Hmade.
    destruct (Typing_env.is_bottom _); [ discriminate Hmade | ].
    injection Hmade as Hmade _. apply combine_new in Hmade. subst made.
    cbn [fst snd]. split; [ exact HQa0 | split; [ | reflexivity ] ].
    exact
      (extract_has QB rb b1 b2 env_b _ HQb1 HQb2 (Hanswers_b scoped) Hmeet_b).
  - (* The first alternative has an answer. *)
    pose proof (Hkept_a ra env_a eq_refl) as Henv_a.
    set (scoped_b :=
      Meet_env.with_names_to_avoid scoped (Meet_env.names_to_avoid env_a))
      in Hmade.
    pose proof (Hb scoped_b) as Hkept_b.
    destruct (meet_b scoped_b b1 b2) as [[rb | rb env_b] | ] eqn:Hmeet_b;
      cbv beta iota in Hmade; [ | | discriminate Hmade ].
    + rewrite (Hcut env_a Henv_a) in Hmade. cbv beta iota in Hmade.
      rewrite add_defined_vars_empty, extension_of_empty_level,
        add_empty_extension_strict in Hmade.
      cbv beta iota in Hmade.
      destruct (Typing_env.is_bottom _); [ discriminate Hmade | ].
      injection Hmade as Hmade _. apply combine_new in Hmade. subst made.
      cbn [fst snd]. split; [ | split; [ exact HQb0 | reflexivity ] ].
      exact
        (extract_has QA ra a1 a2 env_a _ HQa1 HQa2 (Hanswers_a scoped)
           Hmeet_a).
    + pose proof (Hkept_b rb env_b eq_refl) as Henv_b.
      destruct
        (Hjoin
           (Meet_env.with_names_to_avoid env (Meet_env.names_to_avoid env_b))
           [Meet_env.final_typing_env env_a; Meet_env.final_typing_env env_b])
        as (joined & _ & Hjoined);
        [ discriminate
        | intros scoped_env [<- | [<- | []]];
            [ exact Henv_a | exact Henv_b ]
        | ].
      assert (Hjoined' :
        join mh (Typing_env.current_scope (Meet_env.typing_env env))
          (Meet_env.with_names_to_avoid env (Meet_env.names_to_avoid env_b))
          (Meet_env.typing_env env)
          [Meet_env.final_typing_env env_a; Meet_env.final_typing_env env_b]
        = Some joined) by exact (Hjoined mh).
      rewrite Hjoined' in Hmade. unfold Meet_env.final_typing_env in Hmade.
      rewrite (Hcut env_a Henv_a), (Hcut env_b Henv_b) in Hmade.
      cbv beta iota in Hmade. rewrite !add_defined_vars_empty in Hmade.
      cbv beta iota in Hmade.
      change (Typing_env_level.is_empty Typing_env_level.empty) with true
        in Hmade.
      cbn [andb] in Hmade. cbv beta iota in Hmade.
      injection Hmade as Hmade _. apply combine_new in Hmade. subst made.
      cbn [fst snd]. split; [ | split; [ | reflexivity ] ].
      * exact
          (extract_has QA ra a1 a2 env_a _ HQa1 HQa2 (Hanswers_a scoped)
             Hmeet_a).
      * exact
          (extract_has QB rb b1 b2 env_b _ HQb1 HQb2 (Hanswers_b scoped_b)
             Hmeet_b).
Qed.

(* ======================================================================== *)
(* Row-like structures                                                      *)
(* ======================================================================== *)

Section Row_like.

Context {lattice shape maps_to row_tag known : Type}.

Let case : Type := row_like_case lattice shape maps_to env_extension.

Variable equal_index : lattice -> lattice -> bool.
Variable subset_index : lattice -> lattice -> bool.
Variable union_index : lattice -> lattice -> lattice.
Variable meet_shape : shape -> shape -> Or_bottom.t shape.
Variable is_empty_map_known : known -> bool.
Variable get_singleton_map_known : known -> option (row_tag * case).
Variable merge_map_known :
  (row_tag -> option (Or_unknown.t case) -> option (Or_unknown.t case) ->
   row_like_state -> option (option (Or_unknown.t case) * row_like_state)) ->
  known -> known -> row_like_state -> option (known * row_like_state).

(* The meet of the indices of two cases, as it is written in
   [meet_row_like]. *)
Definition row_meet_index (env : Meet_env.t)
    (i1 i2 : row_like_index lattice shape)
    : Meet_result.t (row_like_index lattice shape) :=
  match meet_shape (index_shape i1) (index_shape i2) with
  | Or_bottom.Bottom => Meet_result.Bottom (New_result tt)
  | Or_bottom.Ok shape =>
    match index_domain i1, index_domain i2 with
    | Row_like_index_domain.Known i1', Row_like_index_domain.Known i2' =>
      if equal_index i1' i2'
      then Meet_result.Ok Both_inputs env
      else Meet_result.Bottom (New_result tt)
    | Row_like_index_domain.Known known,
      Row_like_index_domain.At_least at_least =>
      if subset_index at_least known
      then Meet_result.Ok Left_input env
      else Meet_result.Bottom (New_result tt)
    | Row_like_index_domain.At_least at_least,
      Row_like_index_domain.Known known =>
      if subset_index at_least known
      then Meet_result.Ok Right_input env
      else Meet_result.Bottom (New_result tt)
    | Row_like_index_domain.At_least i1',
      Row_like_index_domain.At_least i2' =>
      if subset_index i1' i2'
      then
        if subset_index i2' i1'
        then Meet_result.Ok Both_inputs env
        else Meet_result.Ok Right_input env
      else if subset_index i2' i1'
      then Meet_result.Ok Left_input env
      else
        let domain :=
          Row_like_index_domain.At_least (union_index i1' i2')
        in
        Meet_result.Ok (New_result (Row_like_index domain shape)) env
    end
  end.

(* What [meet_row_like] does with the cases of one tag: its local
   functions, as they are written there, with what they take of the
   function around them as parameters. A meet of two cases that are
   known is the last of its alternatives. *)
Definition row_meet_knowns (mh0 : meet_expanded_head)
    (meet_maps_to0 :
      Meet_env.t -> maps_to -> maps_to -> option (Meet_result.t maps_to))
    (need_join : bool) (common_scope : Scope.t) (base_env : Meet_env.t)
    (other1 other2 : Or_bottom.t case)
    (case1 case2 : option (Or_unknown.t case)) (state : row_like_state)
    : option (option (Or_unknown.t case) * row_like_state) :=
  let base_tenv := Meet_env.final_typing_env base_env in
  let add_extra_variables_and_extract_extension (scoped_env : Typing_env.t)
      (state : row_like_state) : option (env_extension * row_like_state) :=
    let* level := Typing_env.cut scoped_env common_scope in
    let* extra :=
      union_extra_variables (extra_variables state)
        (rev (Typing_env_level.defined_variables_with_kinds level))
    in
    Some
      (extension_of_level level,
       {| extra_variables := extra;
          result_env := result_env state;
          result_is_t1 := result_is_t1 state;
          result_is_t2 := result_is_t2 state;
          avoid := avoid state |})
  in
  (* [join_result_env], with the names to avoid of the environment that
     the scoped one came from. [None] is the assertion of the compiler
     that only one environment is kept unless a join is needed. *)
  let join_result_env (scoped_env : Typing_env.t) (names : list variable)
      (state : row_like_state) : option row_like_state :=
    let* new_result_env :=
      match result_env state with
      | None => Some [scoped_env]
      | Some other_envs =>
        if need_join then Some (scoped_env :: other_envs) else None
      end
    in
    Some
      {| extra_variables := extra_variables state;
         result_env := Some new_result_env;
         result_is_t1 := result_is_t1 state;
         result_is_t2 := result_is_t2 state;
         avoid := names |}
  in
  let bottom_case {A : Type} (r : meet_return_value A)
      (state : row_like_state)
      : option (option (Or_unknown.t case) * row_like_state) :=
    Some (None, update_refs r state)
  in
  (* The environment of a case to which an extension is added. *)
  let add_case_extension (extension : env_extension) (state : row_like_state)
      : option (Or_bottom.t Meet_env.t) :=
    let* added :=
      Meet_env.add_env_extension_strict
        (Meet_env.with_names_to_avoid base_env (avoid state))
        extension mh0
    in
    match added with
    | Or_bottom.Bottom => Some Or_bottom.Bottom
    | Or_bottom.Ok env =>
      match Meet_env.final_typing_env_strict env with
      | Or_bottom.Bottom => Some Or_bottom.Bottom
      | Or_bottom.Ok _ => Some (Or_bottom.Ok env)
      end
    end
  in
  let meet_case (env : Meet_env.t) (case1 case2 : case)
      (state : row_like_state)
      : option (option (Or_unknown.t case) * row_like_state) :=
    let env := Meet_env.with_names_to_avoid env (avoid state) in
    match row_meet_index env (case_index case1) (case_index case2) with
    | Meet_result.Bottom r => bottom_case r state
    | Meet_result.Ok index_result env =>
      let* maps_to :=
        meet_maps_to0 env (case_maps_to case1) (case_maps_to case2)
      in
      match maps_to with
      | Meet_result.Bottom r => bottom_case r state
      | Meet_result.Ok maps_to_result env =>
        let* env :=
          Meet_env.add_env_extension_strict env (case_env_extension case1)
            mh0
        in
        let* env :=
          match env with
          | Or_bottom.Bottom => Some Or_bottom.Bottom
          | Or_bottom.Ok env =>
            Meet_env.add_env_extension_strict env (case_env_extension case2)
              mh0
          end
        in
        let env :=
          match env with
          | Or_bottom.Bottom => Or_bottom.Bottom
          | Or_bottom.Ok env =>
            match Meet_env.final_typing_env_strict env with
            | Or_bottom.Bottom => Or_bottom.Bottom
            | Or_bottom.Ok _ => Or_bottom.Ok env
            end
          end
        in
        match env with
        | Or_bottom.Bottom => bottom_case (@New_result unit tt) state
        | Or_bottom.Ok env =>
          let tenv := Meet_env.final_typing_env env in
          let* state :=
            join_result_env tenv (Meet_env.names_to_avoid env) state
          in
          let state := update_refs index_result state in
          let state := update_refs maps_to_result state in
          let index :=
            extract_value index_result (case_index case1) (case_index case2)
          in
          let maps_to :=
            extract_value maps_to_result (case_maps_to case1)
              (case_maps_to case2)
          in
          (* Not in the compiler: where no extension is made of the
             case, its variables may be lifted all the same (see
             [meet_choices]). *)
          let* '(env_extension, state) :=
            if need_join
            then add_extra_variables_and_extract_extension tenv state
            else if lift_variables_of_single_case choices
            then
              let* '(_, state) :=
                add_extra_variables_and_extract_extension tenv state
              in
              Some (empty_env_extension, state)
            else Some (empty_env_extension, state)
          in
          let state :=
            match env_extension with
            | Env_extension [] => state
            | Env_extension (_ :: _) => not_t1 (not_t2 state)
            end
          in
          let case := Row_like_case maps_to index env_extension in
          Some (Some (Or_unknown.Known case), state)
        end
      end
    end
  in
  (* A case of one input that is unknown, with a case of the other: the
     case of the other, if its extension has a value. *)
  let meet_case_with_unknown (other_case : case)
      (change : row_like_state -> row_like_state) (state : row_like_state)
      : option (option (Or_unknown.t case) * row_like_state) :=
    let* env := add_case_extension (case_env_extension other_case) state in
    match env with
    | Or_bottom.Bottom => Some (None, state)
    | Or_bottom.Ok env =>
      let* state :=
        join_result_env (Meet_env.final_typing_env env)
          (Meet_env.names_to_avoid env) state
      in
      Some (Some (Or_unknown.Known other_case), change state)
    end
  in
  match case1, case2 with
  | None, None => Some (None, state)
  | Some case1, None =>
    match other2 with
    | Or_bottom.Bottom => Some (None, not_t1 state)
    | Or_bottom.Ok other_case =>
      match case1 with
      | Or_unknown.Unknown =>
        meet_case_with_unknown other_case
          (fun state => not_t1 (not_t2 state))
          state
      | Or_unknown.Known case1 => meet_case base_env case1 other_case state
      end
    end
  | None, Some case2 =>
    match other1 with
    | Or_bottom.Bottom => Some (None, not_t2 state)
    | Or_bottom.Ok other_case =>
      match case2 with
      | Or_unknown.Unknown =>
        meet_case_with_unknown other_case
          (fun state => not_t1 (not_t2 state))
          state
      | Or_unknown.Known case2 => meet_case base_env other_case case2 state
      end
    end
  | Some case1, Some case2 =>
    match case1, case2 with
    | Or_unknown.Unknown, Or_unknown.Unknown =>
      let* state := join_result_env base_tenv (avoid state) state in
      Some (Some Or_unknown.Unknown, state)
    | Or_unknown.Known case, Or_unknown.Unknown =>
      meet_case_with_unknown case not_t2 state
    | Or_unknown.Unknown, Or_unknown.Known case =>
      meet_case_with_unknown case not_t1 state
    | Or_unknown.Known case1, Or_unknown.Known case2 =>
      meet_case base_env case1 case2 state
    end
  end.

Lemma meet_row_like_unfold (mh0 : meet_expanded_head)
    (meet_maps_to0 :
      Meet_env.t -> maps_to -> maps_to -> option (Meet_result.t maps_to))
    (initial_env : Meet_env.t) (known1 known2 : known)
    (other1 other2 : Or_bottom.t case) :
  meet_row_like join choices mh0 meet_maps_to0 equal_index subset_index
    union_index meet_shape is_empty_map_known get_singleton_map_known
    merge_map_known initial_env known1 known2 other1 other2
  = let common_scope :=
      Typing_env.current_scope (Meet_env.typing_env initial_env)
    in
    let initial_tenv := Meet_env.typing_env initial_env in
    let base_env :=
      {| Meet_env.typing_env := Typing_env.increment_scope initial_tenv;
         Meet_env.adding_equations_for_names :=
           Meet_env.adding_equations_for_names initial_env;
         Meet_env.names_to_avoid := Meet_env.names_to_avoid initial_env |}
    in
    let extract_and_join_extensions (scoped_envs : list Typing_env.t)
        (state : row_like_state) : option Meet_env.t :=
      let* result_env :=
        join mh0 common_scope
          (Meet_env.with_names_to_avoid initial_env (avoid state))
          initial_tenv scoped_envs
      in
      List.fold_left
        (fun env binding =>
           let* env := env in
           define_in_types choices env (fst binding) (snd binding))
        (extra_variables state) (Some result_env)
    in
    let need_join : bool :=
      match
        other1, get_singleton_map_known known1, other2,
        get_singleton_map_known known2
      with
      | Or_bottom.Bottom, Some _, _, _ | _, _, Or_bottom.Bottom, Some _ => false
      | (Or_bottom.Ok _ | Or_bottom.Bottom), _,
        (Or_bottom.Ok _ | Or_bottom.Bottom), _ =>
        if is_empty_map_known known1 && is_empty_map_known known2
        then false
        else true
      end
    in
    let knowns :=
      row_meet_knowns mh0 meet_maps_to0 need_join common_scope base_env
        other1 other2
    in
    let state :=
      {| extra_variables := [];
         result_env := None;
         result_is_t1 := true;
         result_is_t2 := true;
         avoid := Meet_env.names_to_avoid base_env |}
    in
    let* '(known, state) :=
      merge_map_known (fun _tag case1 case2 => knowns case1 case2) known1
        known2 state
    in
    let* '(other, state) :=
      match other1, other2 with
      | Or_bottom.Bottom, Or_bottom.Bottom => Some (Or_bottom.Bottom, state)
      | Or_bottom.Bottom, _ => Some (Or_bottom.Bottom, not_t2 state)
      | _, Or_bottom.Bottom => Some (Or_bottom.Bottom, not_t1 state)
      | Or_bottom.Ok other1, Or_bottom.Ok other2 =>
        let* '(other, state) :=
          knowns (Some (Or_unknown.Known other1))
            (Some (Or_unknown.Known other2)) state
        in
        match other with
        | None => Some (Or_bottom.Bottom, state)
        | Some Or_unknown.Unknown => None
        | Some (Or_unknown.Known r) => Some (Or_bottom.Ok r, state)
        end
      end
    in
    if is_empty_map_known known
       && match other with
          | Or_bottom.Bottom => true
          | Or_bottom.Ok _ => false
          end
    then Some (Meet_result.Bottom (New_result tt))
    else
      let* env :=
        match result_env state with
        | None => Some Or_bottom.Bottom
        | Some scoped_envs =>
          let* env := extract_and_join_extensions scoped_envs state in
          Some (Or_bottom.Ok env)
        end
      in
      let match_with_input {A : Type} (v : A) : meet_return_value A :=
        match result_is_t1 state, result_is_t2 state with
        | true, true => Both_inputs
        | true, false => Left_input
        | false, true => Right_input
        | false, false => New_result v
        end
      in
      match env with
      | Or_bottom.Bottom => Some (Meet_result.Bottom (match_with_input tt))
      | Or_bottom.Ok env =>
        Some (Meet_result.Ok (match_with_input (known, other)) env)
      end.
Proof. reflexivity. Qed.

Variable Pmaps1 Pmaps2 : maps_to -> Prop.
Variable meet_maps_to meet_maps_to' :
  Meet_env.t -> maps_to -> maps_to -> option (Meet_result.t maps_to).
Hypothesis Hmaps : steady Pmaps1 Pmaps2 meet_maps_to meet_maps_to'.

(* A case whose contents have [P] and that has no extension. *)
Definition case_has (P : maps_to -> Prop) (c : case) : Prop :=
  P (case_maps_to c) ∧ case_env_extension c = Env_extension [].

Variable initial_env : Meet_env.t.

Let scoped : Meet_env.t :=
  {| Meet_env.typing_env :=
       Typing_env.increment_scope (Meet_env.typing_env initial_env);
     Meet_env.adding_equations_for_names :=
       Meet_env.adding_equations_for_names initial_env;
     Meet_env.names_to_avoid := Meet_env.names_to_avoid initial_env |}.

Let common_scope : Scope.t :=
  Typing_env.current_scope (Meet_env.typing_env initial_env).

(* The state of [meet_row_like] where nothing has been added to the
   environments of the cases: no variable to define, and for
   environments those that were entered. *)
Definition row_state_kept (state : row_like_state) : Prop :=
  extra_variables state = []
  ∧ ∀ envs : list Typing_env.t,
      result_env state = Some envs ->
      envs <> []
      ∧ ∀ e : Typing_env.t,
          In e envs ->
          e = Typing_env.increment_scope (Meet_env.typing_env initial_env).

Lemma row_meet_index_kept (env env' : Meet_env.t)
    (i1 i2 : row_like_index lattice shape)
    (r : meet_return_value (row_like_index lattice shape)) :
  row_meet_index env i1 i2 = Meet_result.Ok r env' -> env' = env.
Proof.
  unfold row_meet_index.
  repeat
    match goal with
    | |- context [match ?x with _ => _ end] => destruct x
    | |- context [if ?x then _ else _] => destruct x
    end;
    intros [= ]; subst; reflexivity.
Qed.

Lemma update_refs_extra {A : Type} (r : meet_return_value A)
    (state : row_like_state) :
  extra_variables (update_refs r state) = extra_variables state.
Proof. destruct r; reflexivity. Qed.

Lemma update_refs_result_env {A : Type} (r : meet_return_value A)
    (state : row_like_state) :
  result_env (update_refs r state) = result_env state.
Proof. destruct r; reflexivity. Qed.

Lemma update_refs_avoid {A : Type} (r : meet_return_value A)
    (state : row_like_state) :
  avoid (update_refs r state) = avoid state.
Proof. destruct r; reflexivity. Qed.

Lemma update_refs_kept {A : Type} (r : meet_return_value A)
    (state : row_like_state) :
  row_state_kept state -> row_state_kept (update_refs r state).
Proof.
  unfold row_state_kept. rewrite update_refs_extra, update_refs_result_env.
  trivial.
Qed.

(* Two answers with a state are the same, and leave the state as it
   is asked to be. *)
Definition row_answer_kept {C : Type} (o o' : option (C * row_like_state))
    : Prop :=
  o = o'
  ∧ ∀ (c : C) (state' : row_like_state),
      o = Some (c, state') -> row_state_kept state'.

Lemma row_answer_kept_none {C : Type} :
  row_answer_kept (@None (C * row_like_state)) None.
Proof. split; [ reflexivity | intros c state'; discriminate ]. Qed.

Lemma row_answer_kept_some {C : Type} (c : C) (state : row_like_state) :
  row_state_kept state -> row_answer_kept (Some (c, state)) (Some (c, state)).
Proof.
  intros Hstate. split; [ reflexivity | ].
  intros c' state' [= _ <-]. exact Hstate.
Qed.

Lemma row_meet_case_steady (need_join : bool) (other1 other2 : Or_bottom.t case)
    (case1 case2 : case) (state : row_like_state) :
  case_has Pmaps1 case1 -> case_has Pmaps2 case2 -> row_state_kept state ->
  row_answer_kept
    (row_meet_knowns mh meet_maps_to need_join common_scope scoped other1
       other2 (Some (Or_unknown.Known case1)) (Some (Or_unknown.Known case2))
       state)
    (row_meet_knowns mh' meet_maps_to' need_join common_scope scoped other1
       other2 (Some (Or_unknown.Known case1)) (Some (Or_unknown.Known case2))
       state).
Proof.
  intros [Hmaps1 Hext1] [Hmaps2 Hext2] Hstate.
  pose proof Hstate as (Hextra & Henvs).
  unfold row_meet_knowns. cbv zeta. cbv beta iota.
  set (scoped0 := Meet_env.with_names_to_avoid scoped (avoid state)).
  destruct (row_meet_index scoped0 (case_index case1) (case_index case2))
    as [r | index_result env_i] eqn:Hindex;
    [ apply row_answer_kept_some, update_refs_kept, Hstate | ].
  apply row_meet_index_kept in Hindex. subst env_i.
  pose proof (Hmaps scoped0 _ _ Hmaps1 Hmaps2) as Hmeet.
  kept_cases Hmeet env1 Henv1;
    [ apply row_answer_kept_none
    | apply row_answer_kept_some, update_refs_kept, Hstate | ].
  rewrite Hext1, Hext2, !add_empty_extension_strict.
  destruct (Typing_env.is_bottom (Meet_env.typing_env env1)) eqn:Hbottom;
    cbv beta iota;
    [ apply row_answer_kept_some, update_refs_kept, Hstate | ].
  rewrite !add_empty_extension_strict, Hbottom. cbv beta iota.
  unfold Meet_env.final_typing_env_strict, Meet_env.final_typing_env.
  cbv beta iota.
  assert (Hentered :
    Meet_env.typing_env env1
    = Typing_env.increment_scope (Meet_env.typing_env initial_env))
    by exact Henv1.
  assert (Hcut :
    Typing_env.cut (Meet_env.typing_env env1) common_scope
    = Some Typing_env_level.empty)
    by (rewrite Hentered; apply cut_entered).
  rewrite Hcut. cbv beta iota.
  split; [ reflexivity | ].
  intros c state'.
  (* The state once the environment of the case is among those to
     join. *)
  assert (Hadded :
    ∀ (new_result_env : list Typing_env.t) (names : list variable),
      new_result_env <> [] ->
      (∀ e : Typing_env.t,
        In e new_result_env ->
        e = Typing_env.increment_scope (Meet_env.typing_env initial_env)) ->
      row_state_kept
        {| extra_variables := extra_variables state;
           result_env := Some new_result_env;
           result_is_t1 := result_is_t1 state;
           result_is_t2 := result_is_t2 state;
           avoid := names |}).
  { intros new_result_env names Hnonempty Hall. unfold row_state_kept.
    cbn [extra_variables result_env].
    split; [ exact Hextra | ].
    intros envs [= <-]. split; assumption. }
  destruct (result_env state) as [other_envs | ] eqn:Hresult;
    [ destruct need_join eqn:Hneed; [ | discriminate ] | ].
  - cbv beta iota.
    match goal with
    | |- context [update_refs index_result ?added] => set (state0 := added)
    end.
    assert (Hstate0 : row_state_kept state0).
    { apply Hadded; [ discriminate | ].
      intros e [<- | Hin]; [ exact Hentered | ].
      exact (proj2 (Henvs other_envs eq_refl) e Hin). }
    clearbody state0. rewrite !update_refs_extra.
    cbn [Typing_env_level.defined_variables_with_kinds Typing_env_level.empty
         Typing_env_level.defined_vars rev union_extra_variables].
    change (extension_of_level Typing_env_level.empty)
      with (Env_extension []).
    cbv beta iota. intros [= _ <-].
    unfold row_state_kept. cbn [extra_variables result_env].
    rewrite !update_refs_result_env. exact Hstate0.
  - cbv beta iota.
    match goal with
    | |- context [update_refs index_result ?added] => set (state0 := added)
    end.
    assert (Hstate0 : row_state_kept state0).
    { apply Hadded; [ discriminate | ].
      intros e [<- | []]. exact Hentered. }
    clearbody state0. rewrite !update_refs_extra.
    cbn [Typing_env_level.defined_variables_with_kinds Typing_env_level.empty
         Typing_env_level.defined_vars rev union_extra_variables].
    change (extension_of_level Typing_env_level.empty)
      with (Env_extension []).
    unfold empty_env_extension.
    destruct need_join; [ | destruct (lift_variables_of_single_case choices) ];
      cbv beta iota; intros [= _ <-];
      try (apply update_refs_kept, update_refs_kept, Hstate0);
      unfold row_state_kept; cbn [extra_variables result_env];
      rewrite !update_refs_result_env; exact Hstate0.
Qed.

Lemma not_t1_kept (state : row_like_state) :
  row_state_kept state -> row_state_kept (not_t1 state).
Proof. intros Hstate. exact Hstate. Qed.

Lemma not_t2_kept (state : row_like_state) :
  row_state_kept state -> row_state_kept (not_t2 state).
Proof. intros Hstate. exact Hstate. Qed.

(* The state once an environment that was entered is among those to
   join, whatever names it is given to avoid. *)
Lemma row_join_result_env_kept (need_join : bool) (tenv : Typing_env.t)
    (names : list variable) (state state' : row_like_state) :
  tenv = Typing_env.increment_scope (Meet_env.typing_env initial_env) ->
  row_state_kept state ->
  (let* new_result_env :=
     match result_env state with
     | Some other_envs =>
       if need_join then Some (tenv :: other_envs) else None
     | None => Some [tenv]
     end
   in
   Some
     {| extra_variables := extra_variables state;
        result_env := Some new_result_env;
        result_is_t1 := result_is_t1 state;
        result_is_t2 := result_is_t2 state;
        avoid := names |})
  = Some state' ->
  row_state_kept state'.
Proof.
  intros Htenv (Hextra & Henvs).
  destruct (result_env state) as [other_envs | ] eqn:Hresult;
    [ destruct need_join; [ | discriminate ] | ];
    intros [= <-]; unfold row_state_kept;
    cbn [extra_variables result_env];
    (split; [ exact Hextra | ]);
    intros envs [= <-]; (split; [ discriminate | ]).
  - intros e [<- | Hin]; [ exact Htenv | ].
    exact (proj2 (Henvs other_envs eq_refl) e Hin).
  - intros e [<- | []]. exact Htenv.
Qed.

(* A case of one input that is unknown, with a case of the other. *)
Ltac row_with_unknown Hstate Hextension change_kept :=
  unfold row_meet_knowns; cbv zeta; cbv beta iota;
  rewrite Hextension, !add_empty_extension_strict;
  destruct (Typing_env.is_bottom _);
  cbv beta iota;
  [ apply row_answer_kept_some, Hstate | ];
  unfold Meet_env.final_typing_env_strict, Meet_env.final_typing_env;
  cbv beta iota;
  split; [ reflexivity | ];
  let c := fresh "c" in
  let state' := fresh "state'" in
  let Hanswer := fresh "Hanswer" in
  let Hjoined := fresh "Hjoined" in
  intros c state' Hanswer;
  match type of Hanswer with
  | match ?joined with _ => _ end = _ =>
    destruct joined as [state0 | ] eqn:Hjoined; [ | discriminate ]
  end;
  injection Hanswer as _ <-;
  apply change_kept;
  exact (row_join_result_env_kept _ _ _ _ _ eq_refl Hstate Hjoined).

Definition bottom_or {A : Type} (P : A -> Prop) (a : Or_bottom.t A) : Prop :=
  match a with
  | Or_bottom.Ok a => P a
  | Or_bottom.Bottom => True
  end.

Lemma both_not_kept (state : row_like_state) :
  row_state_kept state -> row_state_kept (not_t1 (not_t2 state)).
Proof. intros Hstate. exact Hstate. Qed.

Lemma row_meet_knowns_steady (need_join : bool)
    (other1 other2 : Or_bottom.t case)
    (c1 c2 : option (Or_unknown.t case)) (state : row_like_state) :
  bottom_or (case_has Pmaps1) other1 -> bottom_or (case_has Pmaps2) other2 ->
  opt_has (known_has (case_has Pmaps1)) c1 ->
  opt_has (known_has (case_has Pmaps2)) c2 ->
  row_state_kept state ->
  row_answer_kept
    (row_meet_knowns mh meet_maps_to need_join common_scope scoped other1
       other2 c1 c2 state)
    (row_meet_knowns mh' meet_maps_to' need_join common_scope scoped other1
       other2 c1 c2 state).
Proof.
  intros Hother1 Hother2 Hc1 Hc2 Hstate.
  destruct c1 as [[case1 | ] | ], c2 as [[case2 | ] | ]; cbn in Hc1, Hc2.
  - exact (row_meet_case_steady need_join other1 other2 case1 case2 state
             Hc1 Hc2 Hstate).
  - row_with_unknown Hstate (proj2 Hc1) not_t2_kept.
  - destruct other2 as [other_case | ]; cbn in Hother2.
    + exact (row_meet_case_steady need_join other1 (Or_bottom.Ok other_case)
               case1 other_case state Hc1 Hother2 Hstate).
    + apply row_answer_kept_some, not_t1_kept, Hstate.
  - row_with_unknown Hstate (proj2 Hc2) not_t1_kept.
  - split; [ reflexivity | ].
    intros c state'. unfold row_meet_knowns. cbv zeta. cbv beta iota.
    intros Hanswer.
    match type of Hanswer with
    | match ?joined with _ => _ end = _ =>
      destruct joined as [state0 | ] eqn:Hjoined; [ | discriminate ]
    end.
    injection Hanswer as _ <-.
    exact (row_join_result_env_kept _ _ _ _ _ eq_refl Hstate Hjoined).
  - destruct other2 as [other_case | ]; cbn in Hother2.
    + row_with_unknown Hstate (proj2 Hother2) both_not_kept.
    + apply row_answer_kept_some, not_t1_kept, Hstate.
  - destruct other1 as [other_case | ]; cbn in Hother1.
    + exact (row_meet_case_steady need_join (Or_bottom.Ok other_case) other2
               other_case case2 state Hother1 Hc2 Hstate).
    + apply row_answer_kept_some, not_t2_kept, Hstate.
  - destruct other1 as [other_case | ]; cbn in Hother1.
    + row_with_unknown Hstate (proj2 Hother1) both_not_kept.
    + apply row_answer_kept_some, not_t2_kept, Hstate.
  - apply row_answer_kept_some, Hstate.
Qed.

(* A case that has no extension. *)
Definition no_extension (c : case) : Prop :=
  case_env_extension c = Env_extension [].

(* [row_meet_knowns], unfolded in a hypothesis. *)
Ltac row_reduce Hc :=
  unfold row_meet_knowns in Hc; cbv zeta in Hc; cbv beta iota in Hc.

(* The case that the meet of two cases makes has no extension: nothing
   is added to its environment. *)
Lemma row_meet_case_answers (need_join : bool)
    (other1 other2 : Or_bottom.t case) (case1 case2 : case)
    (state state' : row_like_state) (c : option (Or_unknown.t case)) :
  case_has Pmaps1 case1 -> case_has Pmaps2 case2 -> row_state_kept state ->
  row_meet_knowns mh meet_maps_to need_join common_scope scoped other1
    other2 (Some (Or_unknown.Known case1)) (Some (Or_unknown.Known case2))
    state
  = Some (c, state') ->
  opt_has (known_has no_extension) c.
Proof.
  intros [Hmaps1 Hext1] [Hmaps2 Hext2] Hstate Hc.
  row_reduce Hc.
  set (scoped0 := Meet_env.with_names_to_avoid scoped (avoid state)) in Hc.
  destruct (row_meet_index scoped0 (case_index case1) (case_index case2))
    as [r | index_result env_i] eqn:Hindex;
    [ injection Hc as <- _; exact I | ].
  apply row_meet_index_kept in Hindex. subst env_i.
  destruct (Hmaps scoped0 _ _ Hmaps1 Hmaps2) as [_ Hkept].
  destruct (meet_maps_to scoped0 (case_maps_to case1) (case_maps_to case2))
    as [[r | maps_to_result env1] | ] eqn:Hmeet; cbv beta iota in Hc;
    [ injection Hc as <- _; exact I | | discriminate Hc ].
  pose proof (Hkept maps_to_result env1 eq_refl) as Henv1.
  rewrite Hext1, Hext2, !add_empty_extension_strict in Hc.
  destruct (Typing_env.is_bottom (Meet_env.typing_env env1)) eqn:Hbottom;
    cbv beta iota in Hc; [ injection Hc as <- _; exact I | ].
  rewrite !add_empty_extension_strict, Hbottom in Hc. cbv beta iota in Hc.
  unfold Meet_env.final_typing_env_strict, Meet_env.final_typing_env in Hc.
  cbv beta iota in Hc.
  assert (Hcut :
    Typing_env.cut (Meet_env.typing_env env1) common_scope
    = Some Typing_env_level.empty)
    by (rewrite Henv1; apply cut_entered).
  rewrite Hcut in Hc. cbv beta iota in Hc.
  destruct (result_env state) as [other_envs | ];
    [ destruct need_join; [ | discriminate Hc ] | ]; cbv beta iota in Hc.
  - match type of Hc with
    | context [update_refs index_result ?added] => set (state0 := added) in Hc
    end.
    clearbody state0. rewrite !update_refs_extra in Hc.
    cbn [Typing_env_level.defined_variables_with_kinds Typing_env_level.empty
         Typing_env_level.defined_vars rev union_extra_variables] in Hc.
    change (extension_of_level Typing_env_level.empty) with (Env_extension [])
      in Hc.
    cbv beta iota in Hc. injection Hc as <- _. reflexivity.
  - match type of Hc with
    | context [update_refs index_result ?added] => set (state0 := added) in Hc
    end.
    clearbody state0. rewrite !update_refs_extra in Hc.
    cbn [Typing_env_level.defined_variables_with_kinds Typing_env_level.empty
         Typing_env_level.defined_vars rev union_extra_variables] in Hc.
    change (extension_of_level Typing_env_level.empty) with (Env_extension [])
      in Hc.
    unfold empty_env_extension in Hc.
    destruct need_join; [ | destruct (lift_variables_of_single_case choices) ];
      cbv beta iota in Hc; injection Hc as <- _; reflexivity.
Qed.

(* A case of one input that is unknown, with a case of the other: the
   case of the other. *)
Ltac row_unknown_answers Hc Hextension :=
  row_reduce Hc;
  rewrite Hextension, !add_empty_extension_strict in Hc;
  destruct (Typing_env.is_bottom _);
  cbv beta iota in Hc;
  [ injection Hc as <- _; exact I | ];
  unfold Meet_env.final_typing_env_strict, Meet_env.final_typing_env in Hc;
  cbv beta iota in Hc;
  match type of Hc with
  | match ?joined with _ => _ end = _ =>
    destruct joined; [ | discriminate Hc ]
  end;
  injection Hc as <- _; exact Hextension.

(* A case that is made has no extension, where the cases of the inputs
   have none. *)
Lemma row_meet_knowns_answers (need_join : bool)
    (other1 other2 : Or_bottom.t case)
    (c1 c2 : option (Or_unknown.t case)) (state state' : row_like_state)
    (c : option (Or_unknown.t case)) :
  bottom_or (case_has Pmaps1) other1 -> bottom_or (case_has Pmaps2) other2 ->
  opt_has (known_has (case_has Pmaps1)) c1 ->
  opt_has (known_has (case_has Pmaps2)) c2 ->
  row_state_kept state ->
  row_meet_knowns mh meet_maps_to need_join common_scope scoped other1 other2
    c1 c2 state
  = Some (c, state') ->
  opt_has (known_has no_extension) c.
Proof.
  intros Hother1 Hother2 Hc1 Hc2 Hstate Hc.
  destruct c1 as [[case1 | ] | ], c2 as [[case2 | ] | ]; cbn in Hc1, Hc2.
  - exact
      (row_meet_case_answers need_join other1 other2 case1 case2 state state'
         c Hc1 Hc2 Hstate Hc).
  - row_unknown_answers Hc (proj2 Hc1).
  - destruct other2 as [other_case | ]; cbn in Hother2.
    + exact
        (row_meet_case_answers need_join other1 (Or_bottom.Ok other_case)
           case1 other_case state state' c Hc1 Hother2 Hstate Hc).
    + row_reduce Hc. injection Hc as <- _. exact I.
  - row_unknown_answers Hc (proj2 Hc2).
  - row_reduce Hc.
    match type of Hc with
    | match ?joined with _ => _ end = _ =>
      destruct joined; [ | discriminate Hc ]
    end.
    injection Hc as <- _. exact I.
  - destruct other2 as [other_case | ]; cbn in Hother2.
    + row_unknown_answers Hc (proj2 Hother2).
    + row_reduce Hc. injection Hc as <- _. exact I.
  - destruct other1 as [other_case | ]; cbn in Hother1.
    + exact
        (row_meet_case_answers need_join (Or_bottom.Ok other_case) other2
           other_case case2 state state' c Hother1 Hc2 Hstate Hc).
    + row_reduce Hc. injection Hc as <- _. exact I.
  - destruct other1 as [other_case | ]; cbn in Hother1.
    + row_unknown_answers Hc (proj2 Hother1).
    + row_reduce Hc. injection Hc as <- _. exact I.
  - row_reduce Hc. injection Hc as <- _. exact I.
Qed.

(* What [meet_row_like] asks of the merge that it is given, on a map
   that has [Pknown1] and one that has [Pknown2]: with two functions
   for the cases of a tag that answer the same and keep the state as
   it is asked to be, it answers the same and keeps it so. *)
Variable Pknown1 Pknown2 : known -> Prop.

Definition merge_steady : Prop :=
  ∀ (f f' :
       row_tag -> option (Or_unknown.t case) -> option (Or_unknown.t case) ->
       row_like_state -> option (option (Or_unknown.t case) * row_like_state))
    (known1 known2 : known) (state : row_like_state),
    Pknown1 known1 -> Pknown2 known2 -> row_state_kept state ->
    (∀ (tag : row_tag) (c1 c2 : option (Or_unknown.t case))
       (s : row_like_state),
      opt_has (known_has (case_has Pmaps1)) c1 ->
      opt_has (known_has (case_has Pmaps2)) c2 ->
      row_state_kept s ->
      row_answer_kept (f tag c1 c2 s) (f' tag c1 c2 s)) ->
    merge_map_known f known1 known2 state
    = merge_map_known f' known1 known2 state
    ∧ ∀ (known' : known) (state' : row_like_state),
        merge_map_known f known1 known2 state = Some (known', state') ->
        row_state_kept state'.

Hypothesis Hmerge : merge_steady.

Lemma meet_row_like_steady (known1 known2 : known)
    (other1 other2 : Or_bottom.t case) :
  Pknown1 known1 -> Pknown2 known2 ->
  bottom_or (case_has Pmaps1) other1 -> bottom_or (case_has Pmaps2) other2 ->
  same_kept initial_env
    (meet_row_like join choices mh meet_maps_to equal_index subset_index
       union_index meet_shape is_empty_map_known get_singleton_map_known
       merge_map_known initial_env known1 known2 other1 other2)
    (meet_row_like join choices mh' meet_maps_to' equal_index subset_index
       union_index meet_shape is_empty_map_known get_singleton_map_known
       merge_map_known initial_env known1 known2 other1 other2).
Proof.
  intros Hknown1 Hknown2 Hother1 Hother2. rewrite !meet_row_like_unfold.
  cbv zeta.
  match goal with
  | |- context [row_meet_knowns mh meet_maps_to ?nj _ _ other1 other2] =>
    set (need_join := nj)
  end.
  match goal with
  | |- context [merge_map_known _ known1 known2 ?init] => set (state0 := init)
  end.
  assert (Hstate0 : row_state_kept state0).
  { split; [ reflexivity | ]. intros envs. discriminate. }
  destruct
    (Hmerge
       (fun _ c1 c2 =>
          row_meet_knowns mh meet_maps_to need_join common_scope scoped other1
            other2 c1 c2)
       (fun _ c1 c2 =>
          row_meet_knowns mh' meet_maps_to' need_join common_scope scoped
            other1 other2 c1 c2)
       known1 known2 state0 Hknown1 Hknown2 Hstate0
       (fun tag c1 c2 s Hc1 Hc2 Hs =>
          row_meet_knowns_steady need_join other1 other2 c1 c2 s Hother1
            Hother2 Hc1 Hc2 Hs))
    as [Hsame Hkept].
  change (Typing_env.current_scope (Meet_env.typing_env initial_env))
    with common_scope.
  change
    {| Meet_env.typing_env :=
         Typing_env.increment_scope (Meet_env.typing_env initial_env);
       Meet_env.adding_equations_for_names :=
         Meet_env.adding_equations_for_names initial_env;
       Meet_env.names_to_avoid := Meet_env.names_to_avoid initial_env |}
    with scoped.
  rewrite <- Hsame. clear Hsame.
  destruct (merge_map_known _ known1 known2 state0) as [[known' state1] | ]
    eqn:Hmerged; [ | apply same_kept_none ].
  specialize (Hkept known' state1 eq_refl). cbv beta iota.
  match goal with
  | |- same_kept _ (match ?o with _ => _ end) (match ?o' with _ => _ end) =>
    assert (Hother : row_answer_kept o o');
    [ | destruct Hother as [Hother_eq Hother_kept]; rewrite <- Hother_eq;
        clear Hother_eq;
        destruct o as [[other state2] | ] eqn:Ho; [ | apply same_kept_none ];
        specialize (Hother_kept other state2 eq_refl) ]
  end.
  { destruct other1 as [o1 | ], other2 as [o2 | ]; cbn in Hother1, Hother2.
    - destruct
        (row_meet_case_steady need_join (Or_bottom.Ok o1) (Or_bottom.Ok o2)
           o1 o2 state1 Hother1 Hother2 Hkept) as [Heq Hk].
      rewrite <- Heq. clear Heq.
      destruct (row_meet_knowns mh _ _ _ _ _ _ _ _ state1)
        as [[c state2] | ] eqn:Hc; [ | apply row_answer_kept_none ].
      specialize (Hk c state2 eq_refl).
      destruct c as [[r | ] | ];
        first [ apply row_answer_kept_some; exact Hk
              | apply row_answer_kept_none ].
    - apply row_answer_kept_some, not_t1_kept, Hkept.
    - apply row_answer_kept_some, not_t2_kept, Hkept.
    - apply row_answer_kept_some, Hkept. }
  destruct Hother_kept as (Hextra & Henvs).
  destruct (is_empty_map_known known' && _); [ apply same_kept_bottom | ].
  rewrite Hextra.
  destruct (result_env state2) as [scoped_envs | ];
    [ | apply same_kept_bottom ].
  destruct (Henvs scoped_envs eq_refl) as [Hnonempty Hall].
  destruct
    (Hjoin (Meet_env.with_names_to_avoid initial_env (avoid state2))
       scoped_envs Hnonempty Hall) as (result & Hresult & Hjoined).
  assert (Hjoined' :
    ∀ mh0 : meet_expanded_head,
      join mh0 common_scope
        (Meet_env.with_names_to_avoid initial_env (avoid state2))
        (Meet_env.typing_env initial_env) scoped_envs
      = Some result) by exact Hjoined.
  rewrite !Hjoined'. cbn [List.fold_left]. apply same_kept_ok_at.
  exact Hresult.
Qed.

(* What [meet_row_like] asks of the merge for what it makes: the map
   that is made has [Qknown], where every case that the function for
   the cases of a tag makes has no extension. *)
Variable Qknown : known -> Prop.

Definition merge_answers : Prop :=
  ∀ (f :
       row_tag -> option (Or_unknown.t case) -> option (Or_unknown.t case) ->
       row_like_state -> option (option (Or_unknown.t case) * row_like_state))
    (known1 known2 known' : known) (state state' : row_like_state),
    Pknown1 known1 -> Pknown2 known2 -> row_state_kept state ->
    (∀ (tag : row_tag) (c1 c2 c : option (Or_unknown.t case))
       (s s' : row_like_state),
      opt_has (known_has (case_has Pmaps1)) c1 ->
      opt_has (known_has (case_has Pmaps2)) c2 ->
      row_state_kept s ->
      f tag c1 c2 s = Some (c, s') ->
      row_state_kept s' ∧ opt_has (known_has no_extension) c) ->
    merge_map_known f known1 known2 state = Some (known', state') ->
    Qknown known'.

Hypothesis Hmerge_answers : merge_answers.

(* The scrutinee at the head of a hypothesis, by cases. *)
Ltac case_head H :=
  match type of H with
  | match ?x with _ => _ end = _ => destruct x
  | (if ?b then _ else _) = _ => destruct b
  end.

(* The end of [meet_row_like], once the cases are met. *)
Ltac row_finish Hmade :=
  repeat (case_head Hmade; cbv beta iota in Hmade; try discriminate Hmade);
  injection Hmade as Hmade _;
  repeat (case_head Hmade; cbv beta iota in Hmade; try discriminate Hmade);
  injection Hmade as <-; cbn [fst snd]; split.

Lemma meet_row_like_answers (known1 known2 : known)
    (other1 other2 : Or_bottom.t case) :
  Pknown1 known1 -> Pknown2 known2 ->
  bottom_or (case_has Pmaps1) other1 -> bottom_or (case_has Pmaps2) other2 ->
  answers
    (fun made : known * Or_bottom.t case =>
       Qknown (fst made) ∧ bottom_or no_extension (snd made))
    (meet_row_like join choices mh meet_maps_to equal_index subset_index
       union_index meet_shape is_empty_map_known get_singleton_map_known
       merge_map_known initial_env known1 known2 other1 other2).
Proof.
  intros Hknown1 Hknown2 Hother1 Hother2 made env' Hmade.
  rewrite meet_row_like_unfold in Hmade. cbv zeta in Hmade.
  match type of Hmade with
  | context [row_meet_knowns mh meet_maps_to ?nj _ _ other1 other2] =>
    set (need_join := nj) in Hmade
  end.
  match type of Hmade with
  | context [merge_map_known _ known1 known2 ?init] =>
    set (state0 := init) in Hmade
  end.
  assert (Hstate0 : row_state_kept state0).
  { split; [ reflexivity | ]. intros envs. discriminate. }
  change (Typing_env.current_scope (Meet_env.typing_env initial_env))
    with common_scope in Hmade.
  change
    {| Meet_env.typing_env :=
         Typing_env.increment_scope (Meet_env.typing_env initial_env);
       Meet_env.adding_equations_for_names :=
         Meet_env.adding_equations_for_names initial_env;
       Meet_env.names_to_avoid := Meet_env.names_to_avoid initial_env |}
    with scoped in Hmade.
  destruct (merge_map_known _ known1 known2 state0) as [[known' state1] | ]
    eqn:Hmerged; [ | discriminate Hmade ].
  cbv beta iota in Hmade.
  assert (Hstate1 : row_state_kept state1).
  { exact
      (proj2
         (Hmerge
            (fun _ c1 c2 =>
               row_meet_knowns mh meet_maps_to need_join common_scope scoped
                 other1 other2 c1 c2)
            (fun _ c1 c2 =>
               row_meet_knowns mh' meet_maps_to' need_join common_scope scoped
                 other1 other2 c1 c2)
            known1 known2 state0 Hknown1 Hknown2 Hstate0
            (fun tag c1 c2 s Hc1 Hc2 Hs =>
               row_meet_knowns_steady need_join other1 other2 c1 c2 s Hother1
                 Hother2 Hc1 Hc2 Hs))
         known' state1 Hmerged). }
  assert (HQknown : Qknown known').
  { apply
      (Hmerge_answers
         (fun _ c1 c2 =>
            row_meet_knowns mh meet_maps_to need_join common_scope scoped
              other1 other2 c1 c2)
         known1 known2 known' state0 state1 Hknown1 Hknown2 Hstate0);
      [ | exact Hmerged ].
    intros tag c1 c2 c s s' Hc1 Hc2 Hs Hc. split.
    - exact
        (proj2
           (row_meet_knowns_steady need_join other1 other2 c1 c2 s Hother1
              Hother2 Hc1 Hc2 Hs)
           c s' Hc).
    - exact
        (row_meet_knowns_answers need_join other1 other2 c1 c2 s s' c Hother1
           Hother2 Hc1 Hc2 Hs Hc). }
  destruct other1 as [o1 | ], other2 as [o2 | ]; cbn in Hother1, Hother2;
    cbv beta iota in Hmade.
  - destruct
      (row_meet_knowns mh meet_maps_to need_join common_scope scoped
         (Or_bottom.Ok o1) (Or_bottom.Ok o2) (Some (Or_unknown.Known o1))
         (Some (Or_unknown.Known o2)) state1)
      as [[c state2] | ] eqn:Hc; [ | discriminate Hmade ].
    pose proof
      (row_meet_case_answers need_join _ _ o1 o2 state1 state2 c Hother1
         Hother2 Hstate1 Hc) as Hno.
    cbv beta iota in Hmade.
    destruct c as [[r | ] | ]; cbv beta iota in Hmade;
      [ | discriminate Hmade | ]; row_finish Hmade;
      first [ exact HQknown | exact Hno | exact I ].
  - row_finish Hmade; first [ exact HQknown | exact I ].
  - row_finish Hmade; first [ exact HQknown | exact I ].
  - row_finish Hmade; first [ exact HQknown | exact I ].
Qed.

End Row_like.

(* The merges that [meet_row_like] is given. *)
Lemma merge_fold_failing_steady {K A B C : Type} (initial_env : Meet_env.t)
    (eqb : K -> K -> bool) (keys : list K) (PA : option A -> Prop)
    (PB : option B -> Prop)
    (f f' :
      K -> option A -> option B -> row_like_state ->
      option (option C * row_like_state))
    (m1 : Keyed.finite_map K A) (m2 : Keyed.finite_map K B)
    (init : row_like_state) :
  (∀ key : K, PA (Keyed.lookup eqb key m1)) ->
  (∀ key : K, PB (Keyed.lookup eqb key m2)) ->
  row_state_kept initial_env init ->
  (∀ (key : K) (a : option A) (b : option B) (s : row_like_state),
    PA a -> PB b -> row_state_kept initial_env s ->
    row_answer_kept initial_env (f key a b s) (f' key a b s)) ->
  row_answer_kept initial_env (merge_fold_failing eqb keys f m1 m2 init)
    (merge_fold_failing eqb keys f' m1 m2 init).
Proof.
  intros H1 H2 Hinit Hf. unfold merge_fold_failing, Keyed.merge_fold_on.
  match goal with
  | |- row_answer_kept _ (let '(_, _) := List.fold_left ?g _ _ in _)
         (let '(_, _) := List.fold_left ?g' _ _ in _) =>
    assert (Hstep :
      ∀ (acc : Keyed.finite_map K C * option row_like_state) (key : K),
        (∀ s : row_like_state,
          snd acc = Some s -> row_state_kept initial_env s) ->
        g acc key = g' acc key
        ∧ ∀ s : row_like_state,
            snd (g acc key) = Some s -> row_state_kept initial_env s);
    [ | assert (Hfolds :
          ∀ acc : Keyed.finite_map K C * option row_like_state,
            (∀ s : row_like_state,
              snd acc = Some s -> row_state_kept initial_env s) ->
            List.fold_left g keys acc = List.fold_left g' keys acc
            ∧ ∀ s : row_like_state,
                snd (List.fold_left g keys acc) = Some s ->
                row_state_kept initial_env s) ]
  end.
  - intros [out state] key Hacc. cbn [snd fst] in *.
    destruct state as [s | ];
      [ | split; [ reflexivity | intros s; discriminate ] ].
    destruct (Hf key _ _ s (H1 key) (H2 key) (Hacc s eq_refl)) as [<- Hk].
    destruct (f key _ _ s) as [[c s'] | ]; cbn [snd];
      (split; [ reflexivity | ]);
      [ intros s'' [= <-]; exact (Hk c s' eq_refl) | intros s''; discriminate ].
  - induction keys as [ | key keys IH]; intros acc Hacc;
      [ split; [ reflexivity | exact Hacc ] | ].
    cbn [List.fold_left]. destruct (Hstep acc key Hacc) as [<- Hacc'].
    apply IH, Hacc'.
  - destruct (Hfolds ([], Some init)) as [<- Hkept];
      [ intros s [= <-]; exact Hinit | ].
    destruct (List.fold_left _ keys _) as [out state]. cbn [snd] in Hkept.
    destruct state as [s | ]; cbn [option_map];
      [ apply row_answer_kept_some, Hkept; reflexivity
      | apply row_answer_kept_none ].
Qed.

(* The cases of blocks and of closures. *)
Lemma extension_is_empty_eq (extension : env_extension) :
  extension_is_empty extension = true -> extension = Env_extension [].
Proof. destruct extension as [[ | ]]; [ reflexivity | discriminate ]. Qed.

Lemma block_case_has (inside : flambda_type -> bool)
    (c : row_like_block_case) :
  plain_block_case inside c = true -> case_has (all_have inside) c.
Proof.
  unfold plain_block_case. rewrite andb_true_iff.
  intros [Hfields Hextension]. split.
  - intros ty Hin. rewrite forallb_forall in Hfields. exact (Hfields ty Hin).
  - apply extension_is_empty_eq, Hextension.
Qed.

Definition entry_has (inside : flambda_type -> bool) (entry : closures_entry)
    : Prop :=
  plain_closures_entry inside entry = true.

Lemma blocks_merge_steady (initial_env : Meet_env.t) :
  @merge_steady nat block_shape (list flambda_type) Z
    (Keyed.finite_map Z (Or_unknown.t row_like_block_case))
    (fun merge_case known1 known2 state =>
       merge_fold_failing Z.eqb
         (sort_numbers (Keyed.keys_of_both Z.eqb known1 known2))
         merge_case known1 known2 state)
    (all_have inside1) (all_have inside2) initial_env
    (bindings_have (known_has (case_has (all_have inside1))))
    (bindings_have (known_has (case_has (all_have inside2)))).
Proof.
  intros f f' known1 known2 state Hknown1 Hknown2 Hstate Hf.
  apply
    (merge_fold_failing_steady initial_env Z.eqb _
       (opt_has (known_has (case_has (all_have inside1))))
       (opt_has (known_has (case_has (all_have inside2)))));
    [ intros key; apply keyed_lookup_has, Hknown1
    | intros key; apply keyed_lookup_has, Hknown2
    | exact Hstate
    | intros key a b s Ha Hb Hs; apply Hf; assumption ].
Qed.

Lemma closures_merge_steady (initial_env : Meet_env.t) :
  @merge_steady set_of_closures_contents unit closures_entry function_slot
    (finite_map row_like_closures_case)
    (fun merge_case known1 known2 state =>
       merge_fold_failing Pos.eqb (Keyed.keys_of_both Pos.eqb known1 known2)
         (fun fslot case1 case2 state =>
            let case1 := option_map Or_unknown.Known case1 in
            let case2 := option_map Or_unknown.Known case2 in
            let* '(case, state) := merge_case fslot case1 case2 state in
            match case with
            | None => Some (None, state)
            | Some (Or_unknown.Known case) => Some (Some case, state)
            | Some Or_unknown.Unknown => None
            end)
         known1 known2 state)
    (entry_has inside1) (entry_has inside2) initial_env
    (bindings_have (case_has (entry_has inside1)))
    (bindings_have (case_has (entry_has inside2))).
Proof.
  intros f f' known1 known2 state Hknown1 Hknown2 Hstate Hf.
  apply
    (merge_fold_failing_steady initial_env Pos.eqb _
       (opt_has (case_has (entry_has inside1)))
       (opt_has (case_has (entry_has inside2))));
    [ intros key; apply keyed_lookup_has, Hknown1
    | intros key; apply keyed_lookup_has, Hknown2
    | exact Hstate | ].
  intros key a b s Ha Hb Hs. cbv zeta.
  destruct
    (Hf key (option_map Or_unknown.Known a) (option_map Or_unknown.Known b) s)
    as [<- Hk];
    [ destruct a; exact Ha | destruct b; exact Hb | exact Hs | ].
  destruct (f key _ _ s) as [[c s'] | ]; [ | apply row_answer_kept_none ].
  specialize (Hk c s' eq_refl).
  destruct c as [[c | ] | ];
    first [ apply row_answer_kept_some; exact Hk
          | apply row_answer_kept_none ].
Qed.

(* What a merge that threads the state makes: every component is one
   that the function for a key made. *)
Lemma merge_fold_failing_answers {K A B C : Type} (initial_env : Meet_env.t)
    (eqb : K -> K -> bool) (keys : list K) (PA : option A -> Prop)
    (PB : option B -> Prop) (R : C -> Prop)
    (f :
      K -> option A -> option B -> row_like_state ->
      option (option C * row_like_state))
    (m1 : Keyed.finite_map K A) (m2 : Keyed.finite_map K B)
    (init state' : row_like_state) (out : Keyed.finite_map K C) :
  (∀ key : K, PA (Keyed.lookup eqb key m1)) ->
  (∀ key : K, PB (Keyed.lookup eqb key m2)) ->
  row_state_kept initial_env init ->
  (∀ (key : K) (a : option A) (b : option B) (s s' : row_like_state)
     (c : option C),
    PA a -> PB b -> row_state_kept initial_env s ->
    f key a b s = Some (c, s') ->
    row_state_kept initial_env s' ∧ opt_has R c) ->
  merge_fold_failing eqb keys f m1 m2 init = Some (out, state') ->
  bindings_have R out.
Proof.
  intros H1 H2 Hinit Hf. unfold merge_fold_failing, Keyed.merge_fold_on.
  match goal with
  | |- (let '(_, _) := List.fold_left ?g _ _ in _) = _ -> _ =>
    assert (Hfolds :
      ∀ (keys0 : list K)
        (acc : Keyed.finite_map K C * option row_like_state),
        bindings_have R (fst acc) ->
        (∀ s : row_like_state,
          snd acc = Some s -> row_state_kept initial_env s) ->
        bindings_have R (fst (List.fold_left g keys0 acc)))
  end.
  { induction keys0 as [ | key keys0 IH]; intros [out0 state0] Hout Hacc;
      [ exact Hout | ].
    cbn [List.fold_left]. apply IH; cbn [fst snd] in *.
    - destruct state0 as [s | ]; [ | exact Hout ].
      destruct (f key _ _ s) as [[c s'] | ] eqn:Hfk; [ | exact Hout ].
      destruct
        (Hf key _ _ s s' c (H1 key) (H2 key) (Hacc s eq_refl) Hfk)
        as [_ Hc].
      destruct c as [c | ]; cbn [fst]; [ | exact Hout ].
      intros binding Hin. apply in_app_or in Hin.
      destruct Hin as [Hin | [<- | []]]; [ exact (Hout binding Hin) | ].
      exact Hc.
    - destruct state0 as [s | ]; [ | intros s; discriminate ].
      destruct (f key _ _ s) as [[c s'] | ] eqn:Hfk; cbn [snd];
        [ | intros s''; discriminate ].
      intros s'' [= <-].
      exact
        (proj1 (Hf key _ _ s s' c (H1 key) (H2 key) (Hacc s eq_refl) Hfk)). }
  intros Hmerged.
  specialize (Hfolds keys ([], Some init)).
  destruct (List.fold_left _ keys _) as [out0 state0].
  destruct state0 as [s | ]; cbn [option_map] in Hmerged;
    [ | discriminate Hmerged ].
  injection Hmerged as <- _.
  apply Hfolds; [ intros binding [] | intros s0 [= <-]; exact Hinit ].
Qed.

Lemma blocks_merge_answers (initial_env : Meet_env.t) :
  @merge_answers nat block_shape (list flambda_type) Z
    (Keyed.finite_map Z (Or_unknown.t row_like_block_case))
    (fun merge_case known1 known2 state =>
       merge_fold_failing Z.eqb
         (sort_numbers (Keyed.keys_of_both Z.eqb known1 known2))
         merge_case known1 known2 state)
    (all_have inside1) (all_have inside2) initial_env
    (bindings_have (known_has (case_has (all_have inside1))))
    (bindings_have (known_has (case_has (all_have inside2))))
    (bindings_have
       (known_has
          (fun c : row_like_block_case =>
             case_env_extension c = Env_extension []))).
Proof.
  intros f known1 known2 known' state state' Hknown1 Hknown2 Hstate Hf
    Hmerged.
  exact
    (merge_fold_failing_answers initial_env Z.eqb _
       (opt_has (known_has (case_has (all_have inside1))))
       (opt_has (known_has (case_has (all_have inside2)))) _ f known1 known2
       state state' known'
       (fun key => keyed_lookup_has _ _ _ key Hknown1)
       (fun key => keyed_lookup_has _ _ _ key Hknown2) Hstate
       (fun key a b s s' c Ha Hb Hs Hc => Hf key a b c s s' Ha Hb Hs Hc)
       Hmerged).
Qed.

Lemma closures_merge_answers (initial_env : Meet_env.t) :
  @merge_answers set_of_closures_contents unit closures_entry function_slot
    (finite_map row_like_closures_case)
    (fun merge_case known1 known2 state =>
       merge_fold_failing Pos.eqb (Keyed.keys_of_both Pos.eqb known1 known2)
         (fun fslot case1 case2 state =>
            let case1 := option_map Or_unknown.Known case1 in
            let case2 := option_map Or_unknown.Known case2 in
            let* '(case, state) := merge_case fslot case1 case2 state in
            match case with
            | None => Some (None, state)
            | Some (Or_unknown.Known case) => Some (Some case, state)
            | Some Or_unknown.Unknown => None
            end)
         known1 known2 state)
    (entry_has inside1) (entry_has inside2) initial_env
    (bindings_have (case_has (entry_has inside1)))
    (bindings_have (case_has (entry_has inside2)))
    (bindings_have
       (fun c : row_like_closures_case =>
          case_env_extension c = Env_extension [])).
Proof.
  intros f known1 known2 known' state state' Hknown1 Hknown2 Hstate Hf
    Hmerged.
  refine
    (merge_fold_failing_answers initial_env Pos.eqb _
       (opt_has (case_has (entry_has inside1)))
       (opt_has (case_has (entry_has inside2))) _ _ known1 known2 state state'
       known'
       (fun key => keyed_lookup_has _ _ _ key Hknown1)
       (fun key => keyed_lookup_has _ _ _ key Hknown2) Hstate _ Hmerged).
  intros key a b s s' c Ha Hb Hs Hc. cbv zeta in Hc.
  destruct
    (f key (option_map Or_unknown.Known a) (option_map Or_unknown.Known b) s)
    as [[c0 s0] | ] eqn:Hfk; [ | discriminate Hc ].
  assert (Ha' :
    opt_has (known_has (case_has (entry_has inside1)))
      (option_map Or_unknown.Known a)) by (destruct a; exact Ha).
  assert (Hb' :
    opt_has (known_has (case_has (entry_has inside2)))
      (option_map Or_unknown.Known b)) by (destruct b; exact Hb).
  destruct (Hf key _ _ c0 s s0 Ha' Hb' Hs Hfk) as [Hs0 Hc0].
  destruct c0 as [[c0 | ] | ]; cbv beta iota in Hc;
    [ | discriminate Hc | ]; injection Hc as <- <-; split;
    first [ exact Hs0 | exact Hc0 | exact I ].
Qed.

Lemma meet_alloc_mode_steady :
  steady (fun _ : Alloc_mode.t => True) (fun _ => True)
    (fun env alloc_mode1 alloc_mode2 =>
       Some (meet_alloc_mode env alloc_mode1 alloc_mode2))
    (fun env alloc_mode1 alloc_mode2 =>
       Some (meet_alloc_mode env alloc_mode1 alloc_mode2)).
Proof.
  intros env [ | | ] [ | | ] _ _; apply same_kept_ok.
Qed.

(* What [plain_blocks] says of the cases. *)
Lemma plain_blocks_cases (inside : flambda_type -> bool)
    (known : Keyed.finite_map Z (Or_unknown.t row_like_block_case))
    (other : Or_bottom.t row_like_block_case) (alloc_mode : Alloc_mode.t) :
  plain_blocks inside (Row_like_for_blocks known other alloc_mode) = true ->
  bindings_have (known_has (case_has (all_have inside))) known
  ∧ bottom_or (case_has (all_have inside)) other.
Proof.
  cbn [plain_blocks]. rewrite andb_true_iff. intros [Hknown Hother]. split.
  - intros [tag [c | ]] Hin; cbn; [ | exact I ].
    rewrite forallb_forall in Hknown. apply block_case_has.
    exact (Hknown _ Hin).
  - destruct other as [c | ]; cbn; [ apply block_case_has, Hother | exact I ].
Qed.

Lemma meet_row_like_for_blocks_steady :
  steady (fun blocks => plain_blocks inside1 blocks = true)
    (fun blocks => plain_blocks inside2 blocks = true)
    (meet_row_like_for_blocks join choices mh)
    (meet_row_like_for_blocks join choices mh').
Proof.
  intros env [known1 other1 alloc_mode1] [known2 other2 alloc_mode2] H1 H2.
  unfold meet_row_like_for_blocks. cbv zeta.
  apply
    (combine_results2_steady
       (fun known_and_other :
              Keyed.finite_map Z (Or_unknown.t row_like_block_case)
              * Or_bottom.t row_like_block_case =>
          bindings_have (known_has (case_has (all_have inside1)))
            (fst known_and_other)
          ∧ bottom_or (case_has (all_have inside1)) (snd known_and_other))
       (fun known_and_other :
              Keyed.finite_map Z (Or_unknown.t row_like_block_case)
              * Or_bottom.t row_like_block_case =>
          bindings_have (known_has (case_has (all_have inside2)))
            (fst known_and_other)
          ∧ bottom_or (case_has (all_have inside2)) (snd known_and_other))
       (fun _ => True) (fun _ => True));
    try exact I;
    try (apply (plain_blocks_cases _ _ _ _ H1));
    try (apply (plain_blocks_cases _ _ _ _ H2)).
  - intros env' left right [Hknown_l Hother_l] [Hknown_r Hother_r].
    apply
      (meet_row_like_steady _ _ _ _ _ _ _ (all_have inside1) (all_have inside2)
         _ _ meet_int_indexed_product_steady env'
         (bindings_have (known_has (case_has (all_have inside1))))
         (bindings_have (known_has (case_has (all_have inside2))))
         (blocks_merge_steady env'));
      assumption.
  - exact meet_alloc_mode_steady.
Qed.

(* What is made of blocks. Blocks whose cases have no extension are
   plain, whatever is asked of their fields. *)
Lemma forallb_true {A : Type} (l : list A) : forallb (fun _ => true) l = true.
Proof. induction l as [ | a l IH]; [ reflexivity | exact IH ]. Qed.

Lemma no_extension_blocks
    (known : Keyed.finite_map Z (Or_unknown.t row_like_block_case))
    (other : Or_bottom.t row_like_block_case) (alloc_mode : Alloc_mode.t) :
  bindings_have
    (known_has
       (fun c : row_like_block_case =>
          case_env_extension c = Env_extension []))
    known ->
  bottom_or
    (fun c : row_like_block_case => case_env_extension c = Env_extension [])
    other ->
  plain_blocks (fun _ => true) (Row_like_for_blocks known other alloc_mode)
  = true.
Proof.
  intros Hknown Hother. cbn [plain_blocks]. apply andb_true_iff. split.
  - apply forallb_forall. intros [tag [c | ]] Hin; cbn [snd];
      [ | reflexivity ].
    pose proof (Hknown _ Hin) as Hext. cbn in Hext.
    unfold plain_block_case. rewrite forallb_true, Hext. reflexivity.
  - destruct other as [c | ]; [ | reflexivity ].
    cbn in Hother. unfold plain_block_case.
    rewrite forallb_true, Hother. reflexivity.
Qed.

Lemma plain_blocks_weaken (inside : flambda_type -> bool)
    (blocks : row_like_for_blocks) :
  plain_blocks inside blocks = true ->
  plain_blocks (fun _ => true) blocks = true.
Proof.
  destruct blocks as [known other alloc_mode]. intros Hplain.
  destruct (plain_blocks_cases inside known other alloc_mode Hplain)
    as [Hknown Hother].
  apply no_extension_blocks.
  - intros [tag [c | ]] Hin; [ | exact I ].
    exact (proj2 (Hknown _ Hin)).
  - destruct other as [c | ]; [ exact (proj2 Hother) | exact I ].
Qed.

Lemma meet_row_like_for_blocks_answers (env : Meet_env.t)
    (blocks1 blocks2 : row_like_for_blocks) :
  plain_blocks inside1 blocks1 = true ->
  plain_blocks inside2 blocks2 = true ->
  answers (fun blocks => plain_blocks (fun _ => true) blocks = true)
    (meet_row_like_for_blocks join choices mh env blocks1 blocks2).
Proof.
  destruct blocks1 as [known1 other1 alloc_mode1],
    blocks2 as [known2 other2 alloc_mode2].
  intros H1 H2. unfold meet_row_like_for_blocks. cbv zeta.
  destruct (plain_blocks_cases _ _ _ _ H1) as [Hknown1 Hother1].
  destruct (plain_blocks_cases _ _ _ _ H2) as [Hknown2 Hother2].
  apply
    (combine_results2_answers
       (fun known_and_other :
              Keyed.finite_map Z (Or_unknown.t row_like_block_case)
              * Or_bottom.t row_like_block_case =>
          bindings_have
            (known_has
               (fun c : row_like_block_case =>
                  case_env_extension c = Env_extension []))
            (fst known_and_other)
          ∧ bottom_or
              (fun c : row_like_block_case =>
                 case_env_extension c = Env_extension [])
              (snd known_and_other))
       (fun _ : Alloc_mode.t => True)).
  - split.
    + intros [tag [c | ]] Hin; [ | exact I ].
      exact (proj2 (Hknown1 _ Hin)).
    + destruct other1 as [c | ]; [ exact (proj2 Hother1) | exact I ].
  - split.
    + intros [tag [c | ]] Hin; [ | exact I ].
      exact (proj2 (Hknown2 _ Hin)).
    + destruct other2 as [c | ]; [ exact (proj2 Hother2) | exact I ].
  - exact I.
  - exact I.
  - exact
      (meet_row_like_answers _ _ _ _ _ _ _ (all_have inside1)
         (all_have inside2) _ _ meet_int_indexed_product_steady env
         (bindings_have (known_has (case_has (all_have inside1))))
         (bindings_have (known_has (case_has (all_have inside2))))
         (blocks_merge_steady env) _ (blocks_merge_answers env) known1 known2
         other1 other2 Hknown1 Hknown2 Hother1 Hother2).
  - intros env1 a env' Ha. exact I.
  - intros [known other] alloc_mode [Hknown Hother] _.
    apply no_extension_blocks; assumption.
Qed.

Lemma closures_case_has (inside : flambda_type -> bool)
    (c : row_like_closures_case) :
  plain_closures_entry inside (case_maps_to c)
  && extension_is_empty (case_env_extension c)
  = true ->
  case_has (entry_has inside) c.
Proof.
  rewrite andb_true_iff. intros [Hentry Hextension].
  split; [ exact Hentry | apply extension_is_empty_eq, Hextension ].
Qed.

Lemma plain_closures_cases (inside : flambda_type -> bool)
    (known : finite_map row_like_closures_case) :
  plain_closures inside (Row_like_for_closures known) = true ->
  bindings_have (case_has (entry_has inside)) known.
Proof.
  intros Hall [slot c] Hin. cbn [plain_closures] in Hall.
  rewrite forallb_forall in Hall. apply closures_case_has.
  exact (Hall _ Hin).
Qed.

Lemma meet_row_like_for_closures_steady :
  steady (fun closures => plain_closures inside1 closures = true)
    (fun closures => plain_closures inside2 closures = true)
    (meet_row_like_for_closures join choices mh)
    (meet_row_like_for_closures join choices mh').
Proof.
  intros env [known1] [known2] H1 H2. unfold meet_row_like_for_closures.
  cbv zeta.
  match goal with
  | |- same_kept _ (match ?m with _ => _ end) (match ?m' with _ => _ end) =>
    assert (Hrow : same_kept env m m')
  end.
  { apply
      (meet_row_like_steady _ _ _ _ _ _ _ (entry_has inside1)
         (entry_has inside2) _ _ meet_closures_entry_steady env
         (bindings_have (case_has (entry_has inside1)))
         (bindings_have (case_has (entry_has inside2)))
         (closures_merge_steady env));
      first [ exact I | apply plain_closures_cases; assumption ]. }
  kept_cases Hrow env1 Henv1.
  - apply same_kept_none.
  - apply same_kept_bottom.
  - destruct r as [ | | | [known' [other | ]]];
      first [ apply same_kept_ok_at, Henv1 | apply same_kept_none ].
Qed.

(* What is made of closures. *)
Lemma no_extension_closures (known : finite_map row_like_closures_case) :
  bindings_have
    (fun c : row_like_closures_case =>
       case_env_extension c = Env_extension [])
    known ->
  plain_closures (fun _ => true) (Row_like_for_closures known) = true.
Proof.
  intros Hknown. cbn [plain_closures]. apply forallb_forall.
  intros [slot c] Hin. pose proof (Hknown _ Hin) as Hext.
  cbn [snd] in Hext |- *. rewrite Hext.
  destruct (case_maps_to c) as [function_types closure_types value_slots].
  cbn [plain_closures_entry]. rewrite !forallb_true. reflexivity.
Qed.

Lemma plain_closures_weaken (inside : flambda_type -> bool)
    (closures : row_like_for_closures) :
  plain_closures inside closures = true ->
  plain_closures (fun _ => true) closures = true.
Proof.
  destruct closures as [known]. intros Hplain.
  apply no_extension_closures. intros binding Hin.
  exact (proj2 (plain_closures_cases inside known Hplain binding Hin)).
Qed.

Lemma meet_row_like_for_closures_answers (env : Meet_env.t)
    (closures1 closures2 : row_like_for_closures) :
  plain_closures inside1 closures1 = true ->
  plain_closures inside2 closures2 = true ->
  answers (fun closures => plain_closures (fun _ => true) closures = true)
    (meet_row_like_for_closures join choices mh env closures1 closures2).
Proof.
  destruct closures1 as [known1], closures2 as [known2].
  intros H1 H2 made env' Hmade.
  unfold meet_row_like_for_closures in Hmade. cbv zeta in Hmade.
  match type of Hmade with
  | match ?m with _ => _ end = _ =>
    assert (Hrow :
      answers
        (fun made0 :
               finite_map row_like_closures_case
               * Or_bottom.t row_like_closures_case =>
           bindings_have
             (fun c : row_like_closures_case =>
                case_env_extension c = Env_extension [])
             (fst made0)
           ∧ bottom_or
               (fun c : row_like_closures_case =>
                  case_env_extension c = Env_extension [])
               (snd made0))
        m)
  end.
  { apply
      (meet_row_like_answers _ _ _ _ _ _ _ (entry_has inside1)
         (entry_has inside2) _ _ meet_closures_entry_steady env
         (bindings_have (case_has (entry_has inside1)))
         (bindings_have (case_has (entry_has inside2)))
         (closures_merge_steady env) _ (closures_merge_answers env));
      first [ exact I | apply plain_closures_cases; assumption ]. }
  match type of Hmade with
  | match ?m with _ => _ end = _ =>
    destruct m as [[r | r env1] | ]; cbv beta iota in Hmade;
      try discriminate Hmade
  end.
  destruct r as [ | | | [known' [other | ]]];
    cbn [map_return_value fst] in Hmade; try discriminate Hmade.
  injection Hmade as <- _. apply no_extension_closures.
  exact (proj1 (Hrow (known', Or_bottom.Bottom) env1 eq_refl)).
Qed.

Lemma set_meet_steady {K B : Type} (eqb : K -> K -> bool)
    (of_set : list K -> B) :
  steady (fun _ : list K => True) (fun _ => True)
    (fun env s1 s2 => Some (set_meet eqb of_set env s1 s2))
    (fun env s1 s2 => Some (set_meet eqb of_set env s1 s2)).
Proof.
  intros env s1 s2 _ _. unfold set_meet.
  destruct (Keyed.all_among eqb s1 s2), (Keyed.all_among eqb s2 s1);
    try apply same_kept_ok.
  destruct (Keyed.is_empty _); [ apply same_kept_bottom | apply same_kept_ok ].
Qed.

Lemma meet_head_of_kind_naked_immediate_steady :
  steady (fun head => plain_naked_immediate head = true)
    (fun head => plain_naked_immediate head = true)
    (meet_head_of_kind_naked_immediate choices mh)
    (meet_head_of_kind_naked_immediate choices mh').
Proof.
  intros env [imms1 relations1'] [imms2 relations2'] H1 H2. cbn in H1, H2.
  destruct relations1'; [ | discriminate ].
  destruct relations2'; [ | discriminate ].
  unfold meet_head_of_kind_naked_immediate. cbn.
  match goal with
  | |- same_kept _ (match ?m with _ => _ end) _ =>
    assert (Himms : same_kept env m m)
  end.
  { apply
      (meet_unknown_steady (fun _ : list Z => True) (fun _ => True) _ _ _
         (set_meet_steady _ _));
      [ destruct imms1; exact I | destruct imms2; exact I ]. }
  apply same_kept_cases in Himms.
  destruct Himms as [_ [-> | [[r ->] | (r & env1 & Henv1 & ->)]]].
  - apply same_kept_none.
  - apply same_kept_bottom.
  - destruct r as [ | | | [[ | i imms'] | ]], imms1 as [imms1 | ],
      imms2 as [imms2 | ];
      cbn;
      first [ apply same_kept_ok_at, Henv1 | apply same_kept_none
            | apply same_kept_bottom ].
Qed.

(* What is made of a head of naked immediates has no inverse relation,
   where the inputs have none. *)
Lemma meet_head_of_kind_naked_immediate_answers (env : Meet_env.t)
    (head1 head2 : head_of_kind_naked_immediate) :
  plain_naked_immediate head1 = true ->
  plain_naked_immediate head2 = true ->
  answers (fun head => plain_naked_immediate head = true)
    (meet_head_of_kind_naked_immediate choices mh env head1 head2).
Proof.
  destruct head1 as [imms1 relations1'], head2 as [imms2 relations2'].
  intros H1 H2 made env' Hmade. cbn in H1, H2.
  destruct relations1'; [ | discriminate H1 ].
  destruct relations2'; [ | discriminate H2 ].
  unfold meet_head_of_kind_naked_immediate in Hmade. cbn in Hmade.
  match type of Hmade with
  | match ?m with _ => _ end = _ =>
    destruct m as [[r | r env1] | ]; cbn in Hmade; try discriminate Hmade
  end.
  destruct r as [ | | | [[ | i imms'] | ]], imms1 as [imms1 | ],
    imms2 as [imms2 | ];
    cbn in Hmade; try discriminate Hmade; injection Hmade as <- _;
    reflexivity.
Qed.

(* A variable for a relation, where the side may have one. *)
Definition allowed (relations : bool) (var : option variable) : Prop :=
  relation_variable_allowed relations var = true.

Lemma meet_relation_steady :
  steady (allowed relations1) (allowed relations2) (meet_relation mh)
    (meet_relation mh').
Proof.
  intros env [var1 | ] [var2 | ] H1 H2; unfold allowed in H1, H2;
    cbn in H1, H2; try apply same_kept_ok.
  rewrite H1, H2 in Hrelations. discriminate Hrelations.
Qed.

Definition blocks_have (inside : flambda_type -> bool)
    (blocks : row_like_for_blocks) : Prop :=
  plain_blocks inside blocks = true.

(* What is asked of the immediates of a variant: of a type inside, or
   that they have no value. *)
Definition has_or_bottom (inside : flambda_type -> bool) (ty : flambda_type)
    : Prop :=
  has inside ty ∨ is_obviously_bottom ty = true.

Lemma meet_variant_steady (env : Meet_env.t)
    (is_int1 get_tag1 is_int2 get_tag2 : option variable)
    (blocks1 blocks2 : Or_unknown.t row_like_for_blocks)
    (imms1 imms2 : Or_unknown.t flambda_type) :
  allowed relations1 is_int1 -> allowed relations1 get_tag1 ->
  allowed relations2 is_int2 -> allowed relations2 get_tag2 ->
  known_has (blocks_have inside1) blocks1 ->
  known_has (has_or_bottom inside1) imms1 ->
  known_has (blocks_have inside2) blocks2 ->
  known_has (has_or_bottom inside2) imms2 ->
  same_kept env
    (meet_variant join choices mh env is_int1 get_tag1 blocks1 imms1 is_int2
       get_tag2 blocks2 imms2 No_extensions No_extensions)
    (meet_variant join choices mh' env is_int1 get_tag1 blocks1 imms1 is_int2
       get_tag2 blocks2 imms2 No_extensions No_extensions).
Proof.
  intros Hint1 Htag1 Hint2 Htag2 Hblocks1 Himms1 Hblocks2 Himms2.
  unfold meet_variant. cbv zeta.
  apply
    (combine_results2_steady (allowed relations1) (allowed relations2)
       (fun rest :
              option variable * Or_unknown.t row_like_for_blocks
              * Or_unknown.t flambda_type * variant_extensions =>
          let '(get_tag, blocks, imms, extensions) := rest in
          allowed relations1 get_tag ∧ known_has (blocks_have inside1) blocks
          ∧ known_has (has_or_bottom inside1) imms
          ∧ extensions = No_extensions)
       (fun rest :
              option variable * Or_unknown.t row_like_for_blocks
              * Or_unknown.t flambda_type * variant_extensions =>
          let '(get_tag, blocks, imms, extensions) := rest in
          allowed relations2 get_tag ∧ known_has (blocks_have inside2) blocks
          ∧ known_has (has_or_bottom inside2) imms
          ∧ extensions = No_extensions));
    try assumption; try (repeat split; assumption).
  - exact meet_relation_steady.
  - intros env' [[[get_tag1' blocks1'] imms1'] extensions1]
      [[[get_tag2' blocks2'] imms2'] extensions2]
      (Hg1 & Hb1 & Hi1 & ->) (Hg2 & Hb2 & Hi2 & ->).
    apply same_kept_map.
    apply
      (meet_disjunction_steady (known_has (has_or_bottom inside1))
         (known_has (has_or_bottom inside2))
         (fun tagged : option variable * Or_unknown.t row_like_for_blocks =>
            allowed relations1 (fst tagged)
            ∧ known_has (blocks_have inside1) (snd tagged))
         (fun tagged : option variable * Or_unknown.t row_like_for_blocks =>
            allowed relations2 (fst tagged)
            ∧ known_has (blocks_have inside2) (snd tagged)));
      try assumption; try (split; assumption).
    + exact (meet_unknown_or_bottom_steady _ _ _ _ _ Hinside).
    + intros env'' [g1 b1] [g2 b2] [Hg1' Hbb1] [Hg2' Hbb2].
      cbn [fst snd] in *.
      apply
        (combine_results2_steady (allowed relations1) (allowed relations2)
           (known_has (blocks_have inside1))
           (known_has (blocks_have inside2)));
        try assumption.
      * exact meet_relation_steady.
      * apply meet_unknown_steady, meet_row_like_for_blocks_steady.
Qed.

(* What is made of a variant is plain: its blocks are, and it has no
   extension. *)
Definition variant_made
    (made :
      option variable * option variable * Or_unknown.t row_like_for_blocks
      * Or_unknown.t flambda_type * variant_extensions) : Prop :=
  known_has (blocks_have (fun _ => true)) (snd (fst (fst made)))
  ∧ snd made = No_extensions.

Lemma known_blocks_weaken (inside : flambda_type -> bool)
    (blocks : Or_unknown.t row_like_for_blocks) :
  known_has (blocks_have inside) blocks ->
  known_has (blocks_have (fun _ => true)) blocks.
Proof.
  destruct blocks as [blocks | ]; [ apply plain_blocks_weaken | trivial ].
Qed.

Lemma meet_variant_answers (env : Meet_env.t)
    (is_int1 get_tag1 is_int2 get_tag2 : option variable)
    (blocks1 blocks2 : Or_unknown.t row_like_for_blocks)
    (imms1 imms2 : Or_unknown.t flambda_type) :
  allowed relations1 is_int1 -> allowed relations1 get_tag1 ->
  allowed relations2 is_int2 -> allowed relations2 get_tag2 ->
  known_has (blocks_have inside1) blocks1 ->
  known_has (has_or_bottom inside1) imms1 ->
  known_has (blocks_have inside2) blocks2 ->
  known_has (has_or_bottom inside2) imms2 ->
  answers variant_made
    (meet_variant join choices mh env is_int1 get_tag1 blocks1 imms1 is_int2
       get_tag2 blocks2 imms2 No_extensions No_extensions).
Proof.
  intros Hint1 Htag1 Hint2 Htag2 Hblocks1 Himms1 Hblocks2 Himms2.
  unfold meet_variant. cbv zeta.
  apply
    (combine_results2_answers (fun _ : option variable => True)
       (fun rest :
              option variable * Or_unknown.t row_like_for_blocks
              * Or_unknown.t flambda_type * variant_extensions =>
          known_has (blocks_have (fun _ => true)) (snd (fst (fst rest)))
          ∧ snd rest = No_extensions));
    try exact I.
  - split; [ exact (known_blocks_weaken _ _ Hblocks1) | reflexivity ].
  - split; [ exact (known_blocks_weaken _ _ Hblocks2) | reflexivity ].
  - intros a env' Ha. exact I.
  - intros env1. apply answers_map.
    assert (Hmade :
      answers
        (fun result :
               Or_unknown.t flambda_type
               * (option variable * Or_unknown.t row_like_for_blocks)
               * variant_extensions =>
           (fun _ : Or_unknown.t flambda_type => True) (fst (fst result))
           ∧ (fun tagged :
                    option variable * Or_unknown.t row_like_for_blocks =>
                known_has (blocks_have (fun _ => true)) (snd tagged))
               (snd (fst result))
           ∧ snd result = No_extensions)
        (meet_disjunction join choices mh
           (meet_unknown (meet mh) is_obviously_bottom)
           (fun (env : Meet_env.t)
              (left right :
                option variable * Or_unknown.t row_like_for_blocks) =>
              combine_results2 env (meet_relation mh)
                (meet_unknown (meet_row_like_for_blocks join choices mh)
                   row_like_for_blocks_is_bottom)
                (fst left) (fst right) (snd left) (snd right)
                (fun get_tag blocks => (get_tag, blocks)))
           (fun _ : unit =>
              Or_unknown.Known (bottom (Naked_number Kind_naked_immediate)))
           (fun _ : unit =>
              (None, Or_unknown.Known row_like_for_blocks_bottom))
           env1 imms1 (get_tag1, blocks1) No_extensions imms2
           (get_tag2, blocks2) No_extensions)).
    { apply
        (meet_disjunction_answers (fun _ : Or_unknown.t flambda_type => True)
           (fun tagged : option variable * Or_unknown.t row_like_for_blocks =>
              known_has (blocks_have (fun _ => true)) (snd tagged)));
        try exact I.
      - intros env2.
        exact
          (proj2
             (meet_unknown_or_bottom_steady _ _ _ _ _ Hinside env2 imms1 imms2
                Himms1 Himms2)).
      - intros env2.
        exact
          (proj2
             (combine_results2_steady (allowed relations1)
                (allowed relations2) (known_has (blocks_have inside1))
                (known_has (blocks_have inside2)) _ _ _ _ env2 get_tag1
                get_tag2 blocks1 blocks2 _ meet_relation_steady
                (meet_unknown_steady _ _ _ _ _
                   meet_row_like_for_blocks_steady)
                Htag1 Htag2 Hblocks1 Hblocks2)).
      - exact (known_blocks_weaken _ _ Hblocks1).
      - exact (known_blocks_weaken _ _ Hblocks2).
      - reflexivity.
      - intros env2 a env' Ha. exact I.
      - intros env2. cbn [fst snd].
        apply
          (combine_results2_answers (fun _ : option variable => True)
             (known_has (blocks_have (fun _ => true))));
          try exact I.
        + exact (known_blocks_weaken _ _ Hblocks1).
        + exact (known_blocks_weaken _ _ Hblocks2).
        + intros a env' Ha. exact I.
        + intros env3. apply meet_unknown_answers.
          intros c1 c2 -> ->.
          exact
            (meet_row_like_for_blocks_answers env3 c1 c2 Hblocks1 Hblocks2).
        + intros get_tag blocks _ Hblocks. exact Hblocks. }
    intros [[imms [get_tag blocks]] extensions] env' Hresult.
    destruct (Hmade _ env' Hresult) as (_ & Hblocks & Hext).
    split; assumption.
  - intros is_int [[[get_tag blocks] imms] extensions] _ Hrest. exact Hrest.
Qed.

Lemma meet_head_of_kind_value_non_null_steady :
  steady (fun head => plain_non_null relations1 inside1 head = true)
    (fun head => plain_non_null relations2 inside2 head = true)
    (meet_head_of_kind_value_non_null join choices mh)
    (meet_head_of_kind_value_non_null join choices mh').
Proof.
  intros env head1 head2 H1 H2. unfold meet_head_of_kind_value_non_null.
  destruct head1
    as [is_int1 imms1 get_tag1 blocks1 extensions1 is_unique1 | alloc_mode1
       | k1 contents1 alloc_mode1 | closures1 alloc_mode1 | strs1],
      head2
    as [is_int2 imms2 get_tag2 blocks2 extensions2 is_unique2 | alloc_mode2
       | k2 contents2 alloc_mode2 | closures2 alloc_mode2 | strs2];
    try apply same_kept_bottom.
  - cbn [plain_non_null] in H1, H2. rewrite !andb_true_iff in H1, H2.
    destruct H1 as [[[[Hint1 Htag1] Himms1] Hblocks1] Hext1].
    destruct H2 as [[[[Hint2 Htag2] Himms2] Hblocks2] Hext2].
    destruct extensions1; [ | discriminate ].
    destruct extensions2; [ | discriminate ].
    apply same_kept_map, meet_variant_steady; try assumption.
    + destruct blocks1; [ exact Hblocks1 | exact I ].
    + destruct imms1; [ apply orb_true_iff, Himms1 | exact I ].
    + destruct blocks2; [ exact Hblocks2 | exact I ].
    + destruct imms2; [ apply orb_true_iff, Himms2 | exact I ].
  - destruct blocks1 as [[known other alloc_mode] | ]; [ | apply same_kept_ok ].
    destruct alloc_mode, alloc_mode2; apply same_kept_ok.
  - destruct blocks2 as [[known other alloc_mode] | ]; [ | apply same_kept_ok ].
    destruct alloc_mode1, alloc_mode; apply same_kept_ok.
  - destruct alloc_mode1, alloc_mode2; apply same_kept_ok.
  - match goal with
    | |- context [if ?same then _ else _] => destruct same
    end;
      [ | apply same_kept_bottom ].
    apply
      (combine_results2_steady (has inside1) (has inside2) (fun _ => True)
         (fun _ => True));
      try exact I; try assumption.
    exact meet_alloc_mode_steady.
  - apply
      (combine_results2_steady
         (fun closures => plain_closures inside1 closures = true)
         (fun closures => plain_closures inside2 closures = true)
         (fun _ => True) (fun _ => True));
      try exact I; try assumption.
    + exact meet_row_like_for_closures_steady.
    + exact meet_alloc_mode_steady.
  - unfold set_meet.
    destruct (Keyed.all_among _ strs1 strs2), (Keyed.all_among _ strs2 strs1);
      try apply same_kept_ok.
    destruct (Keyed.is_empty _);
      [ apply same_kept_bottom | apply same_kept_ok ].
Qed.

(* A head that is plain for a side is plain where variables for
   relations are allowed and nothing is asked of the types inside. *)
Lemma plain_non_null_weaken (relations : bool)
    (inside : flambda_type -> bool) (head : head_of_kind_value_non_null) :
  plain_non_null relations inside head = true ->
  plain_non_null true (fun _ => true) head = true.
Proof.
  destruct head
    as [is_int imms get_tag blocks extensions is_unique | alloc_mode
       | k contents alloc_mode | closures alloc_mode | strs];
    cbn [plain_non_null]; intros Hplain;
    [ | reflexivity | reflexivity | exact (plain_closures_weaken _ _ Hplain)
    | reflexivity ].
  rewrite !andb_true_iff in Hplain |- *.
  destruct Hplain as [[[[_ _] _] Hblocks] Hext].
  split; [ split; [ split; [ split | ] | ] | ].
  - destruct is_int; reflexivity.
  - destruct get_tag; reflexivity.
  - destruct imms; reflexivity.
  - destruct blocks as [blocks | ];
      [ exact (plain_blocks_weaken _ _ Hblocks) | reflexivity ].
  - exact Hext.
Qed.

(* What is made of a head of values that is not null is plain. *)
Lemma meet_head_of_kind_value_non_null_answers (env : Meet_env.t)
    (head1 head2 : head_of_kind_value_non_null) :
  plain_non_null relations1 inside1 head1 = true ->
  plain_non_null relations2 inside2 head2 = true ->
  answers (fun head => plain_non_null true (fun _ => true) head = true)
    (meet_head_of_kind_value_non_null join choices mh env head1 head2).
Proof.
  intros H1 H2. unfold meet_head_of_kind_value_non_null.
  destruct head1
    as [is_int1 imms1 get_tag1 blocks1 extensions1 is_unique1 | alloc_mode1
       | k1 contents1 alloc_mode1 | closures1 alloc_mode1 | strs1],
      head2
    as [is_int2 imms2 get_tag2 blocks2 extensions2 is_unique2 | alloc_mode2
       | k2 contents2 alloc_mode2 | closures2 alloc_mode2 | strs2];
    try (intros made env' Hmade; discriminate Hmade).
  - cbn [plain_non_null] in H1, H2. rewrite !andb_true_iff in H1, H2.
    destruct H1 as [[[[Hint1 Htag1] Himms1] Hblocks1] Hext1].
    destruct H2 as [[[[Hint2 Htag2] Himms2] Hblocks2] Hext2].
    destruct extensions1; [ | discriminate ].
    destruct extensions2; [ | discriminate ].
    apply answers_map.
    intros [[[[is_int get_tag] blocks] imms] extensions] env' Hmade.
    assert (Hvariant :
      variant_made (is_int, get_tag, blocks, imms, extensions)).
    { refine
        (meet_variant_answers env is_int1 get_tag1 is_int2 get_tag2 blocks1
           blocks2 imms1 imms2 Hint1 Htag1 Hint2 Htag2 _ _ _ _ _ env' Hmade).
      - destruct blocks1; [ exact Hblocks1 | exact I ].
      - destruct imms1; [ apply orb_true_iff, Himms1 | exact I ].
      - destruct blocks2; [ exact Hblocks2 | exact I ].
      - destruct imms2; [ apply orb_true_iff, Himms2 | exact I ]. }
    destruct Hvariant as [Hblocks Hext]. cbn [fst snd] in Hblocks, Hext.
    subst extensions. cbn [plain_non_null]. rewrite !andb_true_iff.
    split; [ split; [ split; [ split | ] | ] | ].
    + destruct is_int; reflexivity.
    + destruct get_tag; reflexivity.
    + destruct imms; reflexivity.
    + destruct blocks; [ exact Hblocks | reflexivity ].
    + reflexivity.
  - intros made env' Hmade.
    destruct blocks1 as [[known other alloc_mode] | ];
      [ | discriminate Hmade ].
    destruct alloc_mode, alloc_mode2; cbn in Hmade; try discriminate Hmade;
      injection Hmade as <- _; reflexivity.
  - intros made env' Hmade.
    destruct blocks2 as [[known other alloc_mode] | ];
      [ | discriminate Hmade ].
    destruct alloc_mode1, alloc_mode; cbn in Hmade; try discriminate Hmade;
      injection Hmade as <- _; reflexivity.
  - apply answers_map_some. intros alloc_mode. reflexivity.
  - match goal with
    | |- context [if ?same then _ else _] => destruct same
    end;
      [ | intros made env' Hmade; discriminate Hmade ].
    apply
      (combine_results2_answers (fun _ : flambda_type => True)
         (fun _ : Alloc_mode.t => True));
      try exact I.
    + intros a env' Ha. exact I.
    + intros env1 a env' Ha. exact I.
    + intros contents alloc_mode _ _. reflexivity.
  - apply
      (combine_results2_answers
         (fun closures => plain_closures (fun _ => true) closures = true)
         (fun _ : Alloc_mode.t => True));
      try exact I.
    + exact (plain_closures_weaken _ _ H1).
    + exact (plain_closures_weaken _ _ H2).
    + exact (meet_row_like_for_closures_answers env closures1 closures2 H1 H2).
    + intros env1 a env' Ha. exact I.
    + intros closures alloc_mode Hclosures _. exact Hclosures.
  - apply answers_map_some. intros strs. reflexivity.
Qed.

Definition null_allowed (relations : bool) (is_null : is_null) : Prop :=
  match is_null with
  | Not_null => True
  | Maybe_null var => allowed relations var
  end.

Lemma meet_head_of_kind_value_steady :
  steady (fun head => plain_value relations1 inside1 head = true)
    (fun head => plain_value relations2 inside2 head = true)
    (meet_head_of_kind_value join choices mh)
    (meet_head_of_kind_value join choices mh').
Proof.
  intros env [non_null1 is_null1] [non_null2 is_null2] H1 H2.
  unfold meet_head_of_kind_value. cbv zeta.
  cbn [plain_value] in H1, H2. apply andb_true_iff in H1, H2.
  destruct H1 as [Hnon_null1 Hnull1], H2 as [Hnon_null2 Hnull2].
  apply same_kept_map.
  apply
    (meet_disjunction_steady
       (ok_has (fun head => plain_non_null relations1 inside1 head = true))
       (ok_has (fun head => plain_non_null relations2 inside2 head = true))
       (null_allowed relations1) (null_allowed relations2)).
  - apply meet_or_unknown_or_bottom_steady. intros env' elt1 elt2 Helt1 Helt2.
    apply same_kept_map, meet_head_of_kind_value_non_null_steady; assumption.
  - intros env' [ | var1] [ | var2] Hvar1 Hvar2; try apply same_kept_bottom.
    apply same_kept_map, meet_relation_steady; assumption.
  - destruct non_null1; [ exact I | exact Hnon_null1 | exact I ].
  - destruct non_null2; [ exact I | exact Hnon_null2 | exact I ].
  - destruct is_null1 as [ | var]; [ exact I | exact Hnull1 ].
  - destruct is_null2 as [ | var]; [ exact I | exact Hnull2 ].
Qed.

(* A head of the class of a side. *)
Definition head_plain (relations : bool) (inside : flambda_type -> bool)
    (head : Expanded_type.head) : Prop :=
  match head with
  | Expanded_type.Head_value head => plain_value relations inside head = true
  | Expanded_type.Head_naked_immediate head =>
    plain_naked_immediate head = true
  | Expanded_type.Head_naked_number _ _ | Expanded_type.Head_rec_info _
  | Expanded_type.Head_region _ => True
  end.

Lemma meet_expanded_head0_steady :
  steady (head_plain relations1 inside1) (head_plain relations2 inside2)
    (meet_expanded_head0 join choices mh)
    (meet_expanded_head0 join choices mh').
Proof.
  intros env head1 head2 H1 H2. unfold meet_expanded_head0.
  destruct head1 as [v1 | i1 | k1 n1 | [] | []],
    head2 as [v2 | i2 | k2 n2 | [] | []];
    try apply same_kept_none.
  - apply same_kept_map, meet_head_of_kind_value_steady; assumption.
  - apply same_kept_map, meet_head_of_kind_naked_immediate_steady; assumption.
  - match goal with
    | |- context [if ?same then _ else _] => destruct same
    end;
      [ | apply same_kept_none ].
    unfold meet_head_of_kind_naked_number, set_meet.
    destruct (Keyed.all_among _ n1 n2), (Keyed.all_among _ n2 n1);
      try apply same_kept_ok.
    destruct (Keyed.is_empty _);
      [ apply same_kept_bottom | apply same_kept_ok ].
  - apply same_kept_ok.
  - apply same_kept_ok.
Qed.

(* An expanded type of the class of a side. *)
Definition expanded_plain (relations : bool) (inside : flambda_type -> bool)
    (expanded : Expanded_type.t) : Prop :=
  plain_type relations inside (Expanded_type.to_type expanded) = true.

Lemma plain_expanded_head (relations : bool) (inside : flambda_type -> bool)
    (expanded : Expanded_type.t) :
  expanded_plain relations inside expanded ->
  ok_has (head_plain relations inside) (Expanded_type.descr expanded).
Proof.
  destruct expanded as [kind descr].
  unfold expanded_plain, Expanded_type.to_type.
  cbn [Expanded_type.descr Expanded_type.kind].
  destruct descr as [ | head | ]; cbn [ok_has]; try (intros _; exact I).
  destruct head; cbn; trivial.
Qed.

Lemma meet_expanded_head_step_steady :
  steady (expanded_plain relations1 inside1)
    (expanded_plain relations2 inside2)
    (meet_expanded_head_step join choices mh)
    (meet_expanded_head_step join choices mh').
Proof.
  intros env e1 e2 H1 H2. unfold meet_expanded_head_step.
  apply
    (meet_or_unknown_or_bottom_steady (head_plain relations1 inside1)
       (head_plain relations2 inside2));
    [ exact meet_expanded_head0_steady | | ];
    apply plain_expanded_head; assumption.
Qed.

(* What is made of a head of values is plain. *)
Lemma meet_head_of_kind_value_answers (env : Meet_env.t)
    (head1 head2 : head_of_kind_value) :
  plain_value relations1 inside1 head1 = true ->
  plain_value relations2 inside2 head2 = true ->
  answers (fun head => plain_value true (fun _ => true) head = true)
    (meet_head_of_kind_value join choices mh env head1 head2).
Proof.
  destruct head1 as [non_null1 is_null1], head2 as [non_null2 is_null2].
  intros H1 H2. unfold meet_head_of_kind_value. cbv zeta.
  cbn [plain_value] in H1, H2. apply andb_true_iff in H1, H2.
  destruct H1 as [Hnon_null1 Hnull1], H2 as [Hnon_null2 Hnull2].
  assert (Hok1 :
    ok_has (fun head => plain_non_null relations1 inside1 head = true)
      non_null1)
    by (destruct non_null1; [ exact I | exact Hnon_null1 | exact I ]).
  assert (Hok2 :
    ok_has (fun head => plain_non_null relations2 inside2 head = true)
      non_null2)
    by (destruct non_null2; [ exact I | exact Hnon_null2 | exact I ]).
  apply answers_map.
  match goal with
  | |- answers _ ?disjunction =>
    assert (Hmade :
      answers
        (fun result :
               or_unknown_or_bottom head_of_kind_value_non_null * is_null
               * variant_extensions =>
           ok_has (fun head => plain_non_null true (fun _ => true) head = true)
             (fst (fst result))
           ∧ (fun _ : is_null => True) (snd (fst result))
           ∧ snd result = No_extensions)
        disjunction)
  end.
  { apply
      (meet_disjunction_answers
         (ok_has (fun head => plain_non_null true (fun _ => true) head = true))
         (fun _ : is_null => True));
      try exact I.
    - intros env1.
      exact
        (proj2
           (meet_or_unknown_or_bottom_steady _ _ _ _
              (fun env' elt1 elt2 Helt1 Helt2 =>
                 same_kept_map _ _ _ _
                   (meet_head_of_kind_value_non_null_steady env' elt1 elt2
                      Helt1 Helt2))
              env1 non_null1 non_null2 Hok1 Hok2)).
    - intros env1 r env' Hmeet.
      destruct is_null1 as [ | var1], is_null2 as [ | var2];
        try discriminate Hmeet.
      assert (Hvar1 : allowed relations1 var1) by exact Hnull1.
      assert (Hvar2 : allowed relations2 var2) by exact Hnull2.
      exact
        (proj2
           (same_kept_map Maybe_null env1 _ _
              (meet_relation_steady env1 var1 var2 Hvar1 Hvar2))
           r env' Hmeet).
    - destruct non_null1 as [ | head | ]; [ exact I | | exact I ].
      exact (plain_non_null_weaken _ _ _ Hnon_null1).
    - destruct non_null2 as [ | head | ]; [ exact I | | exact I ].
      exact (plain_non_null_weaken _ _ _ Hnon_null2).
    - intros env1. apply meet_or_unknown_or_bottom_answers.
      intros elt1 elt2 -> ->. apply answers_map.
      exact
        (meet_head_of_kind_value_non_null_answers env1 elt1 elt2 Hnon_null1
           Hnon_null2).
    - intros env1 a env' Ha. exact I. }
  intros [[non_null is_null] extensions] env' Hresult.
  destruct (Hmade _ env' Hresult) as (Hnon_null & _ & _).
  cbn [fst snd] in Hnon_null. cbn [plain_value]. apply andb_true_iff. split.
  - destruct non_null as [ | head | ]; [ reflexivity | | reflexivity ].
    exact Hnon_null.
  - destruct is_null as [ | [var | ]]; reflexivity.
Qed.

(* What is made of a head is plain. *)
Lemma meet_expanded_head0_answers (env : Meet_env.t)
    (head1 head2 : Expanded_type.head) :
  head_plain relations1 inside1 head1 -> head_plain relations2 inside2 head2 ->
  answers (expanded_plain true (fun _ => true))
    (meet_expanded_head0 join choices mh env head1 head2).
Proof.
  intros H1 H2. unfold meet_expanded_head0.
  destruct head1 as [v1 | i1 | k1 n1 | [] | []],
    head2 as [v2 | i2 | k2 n2 | [] | []];
    try (intros made env' Hmade; discriminate Hmade).
  - apply answers_map.
    exact (meet_head_of_kind_value_answers env v1 v2 H1 H2).
  - apply answers_map.
    exact (meet_head_of_kind_naked_immediate_answers env i1 i2 H1 H2).
  - match goal with
    | |- context [if ?same then _ else _] => destruct same
    end;
      [ | intros made env' Hmade; discriminate Hmade ].
    apply answers_map_some. intros numbers. reflexivity.
Qed.

(* What a step of the meet makes is plain: it has no extension. *)
Lemma meet_expanded_head_step_answers (env : Meet_env.t)
    (e1 e2 : Expanded_type.t) :
  expanded_plain relations1 inside1 e1 ->
  expanded_plain relations2 inside2 e2 ->
  answers (expanded_plain true (fun _ => true))
    (meet_expanded_head_step join choices mh env e1 e2).
Proof.
  intros H1 H2. unfold meet_expanded_head_step.
  apply meet_or_unknown_or_bottom_answers. intros head1 head2 Hhead1 Hhead2.
  pose proof (plain_expanded_head _ _ _ H1) as Hplain1.
  pose proof (plain_expanded_head _ _ _ H2) as Hplain2.
  rewrite Hhead1 in Hplain1. rewrite Hhead2 in Hplain2.
  exact (meet_expanded_head0_answers env head1 head2 Hplain1 Hplain2).
Qed.

End Step.

(* ======================================================================== *)
(* The meet of types with no name inside                                    *)
(* ======================================================================== *)

(* The meet of two types that have no name inside within a depth, given
   two meets of expanded types that answer the same on what has none
   within that depth. *)
Lemma meet_steady (depth : nat) (mh mh' : meet_expanded_head) :
  steady (fun e => nameless_expanded_within depth e = true)
    (fun e => nameless_expanded_within depth e = true) mh mh' ->
  steady (has (nameless_within depth)) (has (nameless_within depth))
    (FlambdaMeet.meet mh) (FlambdaMeet.meet mh').
Proof.
  intros Hmh env t1 t2 H1 H2. unfold FlambdaMeet.meet.
  destruct (nameless_cases depth t1 H1)
    as [(e1 & He1 & Hn1) | (c1 & Hc1 & Hn1)],
      (nameless_cases depth t2 H2)
    as [(e2 & He2 & Hn2) | (c2 & Hc2 & Hn2)].
  - rewrite !(meet_type_non_alias env t1 t2 e1 e2 _ He1 He2).
    destruct (negb (kind_eqb (kind t1) (kind t2)));
      [ apply same_kept_none | ].
    apply same_kept_map, Hmh; assumption.
  - rewrite !(meet_type_constant_right env t1 t2 e1 c2 _ He1 Hc2).
    destruct (negb (kind_eqb (kind t1) (kind t2)));
      [ apply same_kept_none | ].
    pose proof (Hmh env e1 (Expanded_type.create_const c2) Hn1 Hn2) as Hmeet.
    kept_cases Hmeet env1 Henv1;
      [ apply same_kept_none | apply same_kept_bottom
      | apply same_kept_ok_at, Henv1 ].
  - rewrite !(meet_type_constant_left env t1 t2 c1 e2 _ Hc1 He2).
    destruct (negb (kind_eqb (kind t1) (kind t2)));
      [ apply same_kept_none | ].
    pose proof (Hmh env e2 (Expanded_type.create_const c1) Hn2 Hn1) as Hmeet.
    kept_cases Hmeet env1 Henv1;
      [ apply same_kept_none | apply same_kept_bottom
      | apply same_kept_ok_at, Henv1 ].
  - rewrite !(meet_type_constants env t1 t2 c1 c2 _ Hc1 Hc2).
    destruct (negb (kind_eqb (kind t1) (kind t2)));
      [ apply same_kept_none | ].
    destruct (simple_equal (Const c1) (Const c2)); [ apply same_kept_ok | ].
    destruct (Typing_env.add_alias _ _ _) as [[ | | ] | ];
      first [ apply same_kept_none | apply same_kept_bottom ].
Qed.

(* ======================================================================== *)
(* The depth is enough                                                      *)
(* ======================================================================== *)

(* By induction on the depth: the meet of that fuel over any meet, and
   the meet of that fuel, answer the same on what has no name inside
   within the depth, with the typing environment given. *)
Lemma nameless_over_steady (join : join_scoped_envs) (choices : meet_choices)
    (mh0 : meet_expanded_head) (depth : nat) :
  join_of_nothing join ->
  steady (fun e => nameless_expanded_within depth e = true)
    (fun e => nameless_expanded_within depth e = true)
    (meet_expanded_head_over join choices mh0 depth)
    (Meet.meet_expanded_head join choices depth).
Proof.
  intros Hjoin. induction depth as [ | depth IH].
  - intros env e1 e2 H1. discriminate H1.
  - cbn [meet_expanded_head_over Meet.meet_expanded_head].
    intros env e1 e2 H1 H2. rewrite nameless_expanded_succ in H1, H2.
    exact
      (meet_expanded_head_step_steady join choices _ _ false false
         (nameless_within depth) (nameless_within depth) Hjoin eq_refl
         (meet_steady depth _ _ IH) env e1 e2 H1 H2).
Qed.

Theorem nameless_fuel_enough : nameless_fuel_enough_statement.
Proof.
  intros join choices depth menv e1 e2 Hjoin H1 H2 mh0.
  exact (proj1 (nameless_over_steady join choices mh0 depth Hjoin menv e1 e2
                  H1 H2)).
Qed.

Theorem nameless_meet_adds_nothing : nameless_meet_adds_nothing_statement.
Proof.
  intros join choices depth menv menv' e1 e2 r Hjoin H1 H2 Hmeet.
  destruct
    (nameless_over_steady join choices (fun _ _ _ => None) depth Hjoin menv e1
       e2 H1 H2) as [Hsame Hkept].
  apply (Hkept r menv'). rewrite Hsame. exact Hmeet.
Qed.

Theorem nameless_types_fuel_enough : nameless_types_fuel_enough_statement.
Proof.
  intros join choices depth tenv t1 t2 Hjoin H1 H2 mh0.
  exact
    (proj1
       (meet_steady depth _ _
          (nameless_over_steady join choices mh0 depth Hjoin)
          (Meet_env.create tenv) t1 t2 H1 H2)).
Qed.

Theorem nameless_types_meet_adds_nothing :
  nameless_types_meet_adds_nothing_statement.
Proof.
  intros join choices depth tenv tenv' t1 t2 ty Hjoin H1 H2. unfold Meet.meet.
  destruct (Typing_env.is_bottom tenv); [ discriminate | ].
  destruct
    (meet_steady depth _ _
       (nameless_over_steady join choices (fun _ _ _ => None) depth Hjoin)
       (Meet_env.create tenv) t1 t2 H1 H2) as [Hsame Hkept].
  unfold FlambdaMeet.meet in Hsame, Hkept. rewrite Hsame in Hkept.
  destruct
    (Meet_env.meet_type (Meet_env.create tenv) t1 t2
       (Meet.meet_expanded_head join choices depth))
    as [[r | r env] | ]; cbv beta iota; try discriminate.
  pose proof (Hkept r env eq_refl) as Henv.
  destruct (is_obviously_bottom _); try discriminate. cbn.
  intros [= _ <-]. exact Henv.
Qed.

Theorem model_meet_nameless : model_meet_nameless_statement.
Proof.
  intros join_choices choices join_depth depth fuel tenv tenv' t1 t2 ty Hfuel
    Hdepth H1 H2.
  pose proof (model_join_of_nothing join_choices join_depth fuel Hfuel)
    as Hjoin.
  pose proof (nameless_within_more _ _ t1 Hdepth H1) as H1'.
  pose proof (nameless_within_more _ _ t2 Hdepth H2) as H2'.
  split.
  - exact (nameless_types_fuel_enough _ choices _ tenv t1 t2 Hjoin H1' H2').
  - exact
      (nameless_types_meet_adds_nothing _ choices _ tenv tenv' t1 t2 ty Hjoin
         H1' H2').
Qed.

(* ======================================================================== *)
(* The meet with a shape                                                    *)
(* ======================================================================== *)

(* A type that says nothing or has no value is not an alias, and its
   expansion says nothing or has no value. *)
Lemma says_nothing_expands (ty : flambda_type) :
  says_nothing_or_no_value ty = true ->
  ∃ e : Expanded_type.t,
    Expanded_type.of_non_alias_type ty = Some e
    ∧ (Expanded_type.descr e = Unknown ∨ Expanded_type.descr e = Bottom).
Proof.
  unfold says_nothing_or_no_value.
  destruct ty as [ty | ty | k ty | ty | ty];
    destruct ty as [ | [head | s] | ]; cbn; try discriminate; intros _;
    eexists; (split; [ reflexivity | ]); cbn; auto.
Qed.

(* A step of the meet, on an expanded type that says nothing or has no
   value and any other: it asks no meet, and answers the environment
   that it was given. *)
Lemma step_of_nothing_left (join : join_scoped_envs) (choices : meet_choices)
    (m m' : meet_expanded_head) (env : Meet_env.t) (e1 e2 : Expanded_type.t) :
  Expanded_type.descr e1 = Unknown ∨ Expanded_type.descr e1 = Bottom ->
  meet_expanded_head_step join choices m' env e1 e2
  = meet_expanded_head_step join choices m env e1 e2
  ∧ (meet_expanded_head_step join choices m env e1 e2
     = Some (Meet_result.Ok Both_inputs env)
     ∨ meet_expanded_head_step join choices m env e1 e2
       = Some (Meet_result.Ok Right_input env)
     ∨ ∃ r : meet_return_value unit,
         meet_expanded_head_step join choices m env e1 e2
         = Some (Meet_result.Bottom r)).
Proof.
  intros Hdescr. unfold meet_expanded_head_step, meet_or_unknown_or_bottom.
  destruct Hdescr as [-> | ->]; destruct (Expanded_type.descr e2);
    (split; [ reflexivity | ]); eauto.
Qed.

Lemma step_of_nothing_right (join : join_scoped_envs) (choices : meet_choices)
    (m m' : meet_expanded_head) (env : Meet_env.t) (e1 e2 : Expanded_type.t) :
  Expanded_type.descr e2 = Unknown ∨ Expanded_type.descr e2 = Bottom ->
  meet_expanded_head_step join choices m' env e1 e2
  = meet_expanded_head_step join choices m env e1 e2
  ∧ (meet_expanded_head_step join choices m env e1 e2
     = Some (Meet_result.Ok Both_inputs env)
     ∨ meet_expanded_head_step join choices m env e1 e2
       = Some (Meet_result.Ok Left_input env)
     ∨ ∃ r : meet_return_value unit,
         meet_expanded_head_step join choices m env e1 e2
         = Some (Meet_result.Bottom r)).
Proof.
  intros Hdescr. unfold meet_expanded_head_step, meet_or_unknown_or_bottom.
  destruct Hdescr as [-> | ->]; destruct (Expanded_type.descr e1);
    (split; [ reflexivity | ]); eauto.
Qed.

(* An equation that says nothing or has no value, on a canonical
   element: the type of a name is looked up, and no equation is
   added. [side] is the input that the meet of types then answers. *)
Lemma equation_of_nothing (join : join_scoped_envs) (choices : meet_choices)
    (m m' : meet_expanded_head) (env : Meet_env.t) (canonical : simple)
    (e : Expanded_type.t) (side : meet_return_value Expanded_type.t) :
  Expanded_type.descr e = Unknown ∨ Expanded_type.descr e = Bottom ->
  same_kept env
    (option_map (map_result Expanded_type.to_type)
       (let* result :=
          Meet_env.add_or_delay_concrete_equation_on_canonical true env
            canonical e (meet_expanded_head_step join choices m)
        in
        Some
          match result with
          | Or_bottom.Ok env0 => Meet_result.Ok side env0
          | Or_bottom.Bottom => Meet_result.Bottom (New_result tt)
          end))
    (option_map (map_result Expanded_type.to_type)
       (let* result :=
          Meet_env.add_or_delay_concrete_equation_on_canonical true env
            canonical e (meet_expanded_head_step join choices m')
        in
        Some
          match result with
          | Or_bottom.Ok env0 => Meet_result.Ok side env0
          | Or_bottom.Bottom => Meet_result.Bottom (New_result tt)
          end)).
Proof.
  intros Hdescr.
  assert (Hname :
    ∀ name : name,
      same_kept env
        (option_map (map_result Expanded_type.to_type)
           (let* result :=
              Meet_env.add_or_delay_concrete_equation_on_canonical_name true
                env name e (meet_expanded_head_step join choices m)
            in
            Some
              match result with
              | Or_bottom.Ok env0 => Meet_result.Ok side env0
              | Or_bottom.Bottom => Meet_result.Bottom (New_result tt)
              end))
        (option_map (map_result Expanded_type.to_type)
           (let* result :=
              Meet_env.add_or_delay_concrete_equation_on_canonical_name true
                env name e (meet_expanded_head_step join choices m')
            in
            Some
              match result with
              | Or_bottom.Ok env0 => Meet_result.Ok side env0
              | Or_bottom.Bottom => Meet_result.Bottom (New_result tt)
              end))).
  { intros name.
    unfold Meet_env.add_or_delay_concrete_equation_on_canonical_name.
    destruct (Keyed.among name_eqb name _);
      [ apply same_kept_map, same_kept_ok | ].
    unfold Meet_env.adding_equation_for_name,
      Meet_env.add_concrete_equation_on_canonical_name.
    cbn [Meet_env.typing_env].
    destruct (Typing_env.find _ name _) as [existing | ];
      [ | apply same_kept_none ].
    cbn beta iota.
    destruct (Expanded_type.of_non_alias_type existing) as [e_existing | ];
      [ | apply same_kept_none ].
    cbn beta iota.
    match goal with
    | |- context [meet_expanded_head_step join choices m ?env1 e e_existing] =>
      destruct
        (step_of_nothing_left join choices m m' env1 e e_existing Hdescr)
        as [-> [-> | [-> | [r ->]]]]
    end;
      cbn beta iota;
      first [ apply same_kept_map, same_kept_ok_at; reflexivity
            | apply same_kept_bottom ]. }
  unfold Meet_env.add_or_delay_concrete_equation_on_canonical.
  destruct canonical as [x | sym | c]; [ apply Hname | apply Hname | ].
  unfold Meet_env.add_concrete_equation_on_const.
  destruct
    (step_of_nothing_left join choices m m' env e
       (Expanded_type.create_const c) Hdescr)
    as [-> [-> | [-> | [r ->]]]];
    cbn beta iota;
    first [ apply same_kept_map, same_kept_ok | apply same_kept_bottom ].
Qed.

(* The meet of a type that says nothing or has no value with any type,
   where the meet of expanded types is a step: it answers the same
   whatever the step is of, with the typing environment given. *)
Lemma meet_of_nothing_steady (join : join_scoped_envs)
    (choices : meet_choices) (m m' : meet_expanded_head) :
  steady (has says_nothing_or_no_value) (has (fun _ => true))
    (FlambdaMeet.meet (meet_expanded_head_step join choices m))
    (FlambdaMeet.meet (meet_expanded_head_step join choices m')).
Proof.
  intros env t1 t2 H1 _.
  destruct (says_nothing_expands t1 H1) as (e1 & He1 & Hdescr).
  unfold FlambdaMeet.meet, Meet_env.meet_type, Meet_env.meet.
  destruct (negb (kind_eqb (kind t1) (kind t2))); [ apply same_kept_none | ].
  unfold Typing_env.get_alias_then_canonical_simple_exn at 1 3.
  rewrite (of_non_alias_type_not_alias t1 e1 He1). cbn beta iota.
  unfold expand_head0 at 1 3.
  rewrite (of_non_alias_type_not_alias t1 e1 He1), He1. cbn beta iota.
  destruct (Typing_env.get_alias_then_canonical_simple_exn _ _ _ t2)
    as [[simple2 | ] | ]; [ | | apply same_kept_none ].
  - exact (equation_of_nothing join choices m m' env simple2 e1 _ Hdescr).
  - destruct (expand_head0 _ t2 None) as [e2 | ]; [ | apply same_kept_none ].
    cbn beta iota.
    destruct (step_of_nothing_left join choices m m' env e1 e2 Hdescr)
      as [-> [-> | [-> | [r ->]]]];
      cbn; first [ apply same_kept_ok | apply same_kept_bottom ].
Qed.

(* The same with the two types the other way round. *)
Lemma meet_with_nothing_steady (join : join_scoped_envs)
    (choices : meet_choices) (m m' : meet_expanded_head) :
  steady (has (fun _ => true)) (has says_nothing_or_no_value)
    (FlambdaMeet.meet (meet_expanded_head_step join choices m))
    (FlambdaMeet.meet (meet_expanded_head_step join choices m')).
Proof.
  intros env t1 t2 _ H2.
  destruct (says_nothing_expands t2 H2) as (e2 & He2 & Hdescr).
  unfold FlambdaMeet.meet, Meet_env.meet_type, Meet_env.meet.
  destruct (negb (kind_eqb (kind t1) (kind t2))); [ apply same_kept_none | ].
  unfold Typing_env.get_alias_then_canonical_simple_exn at 2 4.
  rewrite (of_non_alias_type_not_alias t2 e2 He2). cbn beta iota.
  destruct (Typing_env.get_alias_then_canonical_simple_exn _ _ _ t1)
    as [[simple1 | ] | ]; [ | | apply same_kept_none ]; cbn beta iota.
  - unfold expand_head0.
    rewrite (of_non_alias_type_not_alias t2 e2 He2), He2. cbn beta iota.
    exact (equation_of_nothing join choices m m' env simple1 e2 _ Hdescr).
  - destruct (expand_head0 _ t1 None) as [e1 | ]; [ | apply same_kept_none ].
    cbn beta iota. unfold expand_head0.
    rewrite (of_non_alias_type_not_alias t2 e2 He2), He2. cbn beta iota.
    destruct (step_of_nothing_right join choices m m' env e1 e2 Hdescr)
      as [-> [-> | [-> | [r ->]]]];
      cbn; first [ apply same_kept_ok | apply same_kept_bottom ].
Qed.

(* Two steps of the meet, on a shape and a type that has no extension,
   in either order. *)
Lemma shape_steps_steady (join : join_scoped_envs) (choices : meet_choices)
    (m m' : meet_expanded_head) :
  join_of_nothing join ->
  steady (expanded_plain false says_nothing_or_no_value)
    (expanded_plain true (fun _ => true))
    (meet_expanded_head_step join choices
       (meet_expanded_head_step join choices m))
    (meet_expanded_head_step join choices
       (meet_expanded_head_step join choices m')).
Proof.
  intros Hjoin.
  exact
    (meet_expanded_head_step_steady join choices _ _ false true _ _ Hjoin
       eq_refl (meet_of_nothing_steady join choices m m')).
Qed.

Lemma steps_shape_steady (join : join_scoped_envs) (choices : meet_choices)
    (m m' : meet_expanded_head) :
  join_of_nothing join ->
  steady (expanded_plain true (fun _ => true))
    (expanded_plain false says_nothing_or_no_value)
    (meet_expanded_head_step join choices
       (meet_expanded_head_step join choices m))
    (meet_expanded_head_step join choices
       (meet_expanded_head_step join choices m')).
Proof.
  intros Hjoin.
  exact
    (meet_expanded_head_step_steady join choices _ _ true false _ _ Hjoin
       eq_refl (meet_with_nothing_steady join choices m m')).
Qed.

Theorem shape_fuel_enough : shape_fuel_enough_statement.
Proof.
  intros join choices menv e1 e2 Hjoin H1 H2. split; intros mh0.
  - exact
      (proj1
         (shape_steps_steady join choices mh0 (fun _ _ _ => None) Hjoin menv
            e1 e2 H1 H2)).
  - exact
      (proj1
         (steps_shape_steady join choices mh0 (fun _ _ _ => None) Hjoin menv
            e2 e1 H2 H1)).
Qed.

Theorem shape_meet_adds_nothing : shape_meet_adds_nothing_statement.
Proof.
  intros join choices menv menv' e1 e2 r Hjoin H1 H2. split; intros Hmeet.
  - destruct
      (shape_steps_steady join choices (fun _ _ _ => None)
         (fun _ _ _ => None) Hjoin menv e1 e2 H1 H2) as [_ Hkept].
    exact (Hkept r menv' Hmeet).
  - destruct
      (steps_shape_steady join choices (fun _ _ _ => None)
         (fun _ _ _ => None) Hjoin menv e2 e1 H2 H1) as [_ Hkept].
    exact (Hkept r menv' Hmeet).
Qed.

(* A type that is plain for a side is plain where variables for
   relations are allowed and nothing is asked of the types inside: it
   has no extension. *)
Lemma plain_type_weaken (relations : bool) (inside : flambda_type -> bool)
    (ty : flambda_type) :
  plain_type relations inside ty = true ->
  plain_type true (fun _ => true) ty = true.
Proof.
  destruct ty as [ty | ty | k ty | ty | ty];
    destruct ty as [ | [head | s] | ]; cbn; try trivial.
  destruct head as [non_null is_null]. cbn.
  rewrite !andb_true_iff. intros [Hnon_null His_null]. split.
  - destruct non_null as [ | non_null | ]; try reflexivity.
    exact (plain_non_null_weaken _ _ _ Hnon_null).
  - destruct is_null as [ | [var | ]]; reflexivity.
Qed.

(* What two steps of the meet make of a shape and a type that has no
   extension, in either order. *)
Lemma shape_steps_answers (join : join_scoped_envs) (choices : meet_choices)
    (m : meet_expanded_head) (env : Meet_env.t) (e1 e2 : Expanded_type.t) :
  join_of_nothing join ->
  expanded_plain false says_nothing_or_no_value e1 ->
  expanded_plain true (fun _ => true) e2 ->
  answers (expanded_plain true (fun _ => true))
    (meet_expanded_head_step join choices
       (meet_expanded_head_step join choices m) env e1 e2)
  ∧ answers (expanded_plain true (fun _ => true))
      (meet_expanded_head_step join choices
         (meet_expanded_head_step join choices m) env e2 e1).
Proof.
  intros Hjoin H1 H2. split.
  - exact
      (meet_expanded_head_step_answers join choices _ _ false true _ _ Hjoin
         eq_refl (meet_of_nothing_steady join choices m m) env e1 e2 H1 H2).
  - exact
      (meet_expanded_head_step_answers join choices _ _ true false _ _ Hjoin
         eq_refl (meet_with_nothing_steady join choices m m) env e2 e1 H2 H1).
Qed.

Theorem shape_meet_extension_free : shape_meet_extension_free_statement.
Proof.
  intros join choices menv menv' e1 e2 e Hjoin H1 H2.
  destruct
    (shape_steps_answers join choices (fun _ _ _ => None) menv e1 e2 Hjoin H1
       H2) as [Hleft Hright].
  split; intros Hmeet;
    [ exact (Hleft e menv' Hmeet) | exact (Hright e menv' Hmeet) ].
Qed.

(* One step of the meet, on a bare shape and a type that has no
   extension, in either order: no meet of types is asked. *)
Lemma bare_step_steady (join : join_scoped_envs) (choices : meet_choices)
    (m m' : meet_expanded_head) :
  join_of_nothing join ->
  steady (expanded_plain false (fun _ => false))
    (expanded_plain true (fun _ => true))
    (meet_expanded_head_step join choices m)
    (meet_expanded_head_step join choices m').
Proof.
  intros Hjoin.
  apply
    (meet_expanded_head_step_steady join choices m m' false true _ _ Hjoin
       eq_refl).
  intros env t1 t2 H1. discriminate H1.
Qed.

Lemma step_bare_steady (join : join_scoped_envs) (choices : meet_choices)
    (m m' : meet_expanded_head) :
  join_of_nothing join ->
  steady (expanded_plain true (fun _ => true))
    (expanded_plain false (fun _ => false))
    (meet_expanded_head_step join choices m)
    (meet_expanded_head_step join choices m').
Proof.
  intros Hjoin.
  apply
    (meet_expanded_head_step_steady join choices m m' true false _ _ Hjoin
       eq_refl).
  intros env t1 t2 _ H2. discriminate H2.
Qed.

Theorem bare_shape_shape : bare_shape_shape_statement.
Proof.
  intros ty. apply plain_type_more. intros ty' Hnone. discriminate Hnone.
Qed.

Theorem bare_shape_fuel_enough : bare_shape_fuel_enough_statement.
Proof.
  intros join choices menv e1 e2 Hjoin H1 H2. split; intros mh0.
  - exact
      (proj1
         (bare_step_steady join choices mh0 (fun _ _ _ => None) Hjoin menv e1
            e2 H1 H2)).
  - exact
      (proj1
         (step_bare_steady join choices mh0 (fun _ _ _ => None) Hjoin menv e2
            e1 H2 H1)).
Qed.

Theorem bare_shape_meet_adds_nothing : bare_shape_meet_adds_nothing_statement.
Proof.
  intros join choices menv menv' e1 e2 r Hjoin H1 H2. split; intros Hmeet.
  - destruct
      (bare_step_steady join choices (fun _ _ _ => None) (fun _ _ _ => None)
         Hjoin menv e1 e2 H1 H2) as [_ Hkept].
    exact (Hkept r menv' Hmeet).
  - destruct
      (step_bare_steady join choices (fun _ _ _ => None) (fun _ _ _ => None)
         Hjoin menv e2 e1 H2 H1) as [_ Hkept].
    exact (Hkept r menv' Hmeet).
Qed.

(* What one step of the meet makes of a bare shape and a type that has
   no extension, in either order. *)
Lemma bare_step_answers (join : join_scoped_envs) (choices : meet_choices)
    (m : meet_expanded_head) (env : Meet_env.t) (e1 e2 : Expanded_type.t) :
  join_of_nothing join ->
  expanded_plain false (fun _ => false) e1 ->
  expanded_plain true (fun _ => true) e2 ->
  answers (expanded_plain true (fun _ => true))
    (meet_expanded_head_step join choices m env e1 e2)
  ∧ answers (expanded_plain true (fun _ => true))
      (meet_expanded_head_step join choices m env e2 e1).
Proof.
  intros Hjoin H1 H2. split.
  - apply
      (meet_expanded_head_step_answers join choices m m false true
         (fun _ => false) (fun _ => true) Hjoin eq_refl);
      [ | exact H1 | exact H2 ].
    intros env' t1 t2 Ht1. discriminate Ht1.
  - apply
      (meet_expanded_head_step_answers join choices m m true false
         (fun _ => true) (fun _ => false) Hjoin eq_refl);
      [ | exact H2 | exact H1 ].
    intros env' t1 t2 _ Ht2. discriminate Ht2.
Qed.

Theorem bare_shape_meet_extension_free :
  bare_shape_meet_extension_free_statement.
Proof.
  intros join choices menv menv' e1 e2 e Hjoin H1 H2.
  destruct
    (bare_step_answers join choices (fun _ _ _ => None) menv e1 e2 Hjoin H1 H2)
    as [Hleft Hright].
  split; intros Hmeet;
    [ exact (Hleft e menv' Hmeet) | exact (Hright e menv' Hmeet) ].
Qed.

Theorem relation_shapes : relation_shapes_statement.
Proof.
  repeat (split; [ reflexivity | ]).
  intros tags alloc_mode.
  assert (Hblocks :
    plain_blocks (fun _ => false)
      (Row_like_for_blocks (map (fun tag : Z => (tag, Or_unknown.Unknown)) tags)
         Or_bottom.Bottom alloc_mode)
    = true).
  { cbn [plain_blocks]. apply andb_true_iff. split; [ | reflexivity ].
    induction tags as [ | tag tags IH]; [ reflexivity | exact IH ]. }
  split; unfold bare_shape; cbn -[plain_blocks]; rewrite Hblocks;
    reflexivity.
Qed.

(* ======================================================================== *)
(* An example                                                               *)
(* ======================================================================== *)

Theorem example_depth_fuel : example_depth_fuel_statement.
Proof.
  split; [ reflexivity | split; [ reflexivity | split ] ].
  - eexists. vm_compute. reflexivity.
  - vm_compute. reflexivity.
Qed.

Theorem example_const_fuel : example_const_fuel_statement.
Proof.
  cbv zeta.
  split; [ reflexivity | split; [ reflexivity | split; [ reflexivity | ] ] ].
  split.
  - eexists. vm_compute. reflexivity.
  - vm_compute. reflexivity.
Qed.

Theorem example_shape_fuel : example_shape_fuel_statement.
Proof.
  cbv zeta. split; [ reflexivity | split; [ reflexivity | split ] ].
  - eexists. vm_compute. reflexivity.
  - vm_compute. reflexivity.
Qed.

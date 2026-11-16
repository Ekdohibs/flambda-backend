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


(* Proofs of the statements of FlambdaMeetProductsSpec.v. Nothing here
   needs review. *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypeMeaningProofs FlambdaTypeWellFormedSpec
  FlambdaTypeWellFormedProofs FlambdaBindingTime FlambdaEnvAliases
  FlambdaEnvAliasesSpec FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaTypingEnvKindedSpec FlambdaMeetGrammar FlambdaExpandHead
  FlambdaMeetEnv FlambdaMeet FlambdaMeetSpec FlambdaMeetStepsSpec
  FlambdaMeetStepsProofs FlambdaMeetEnvSpec FlambdaMeetEnvProofs
  FlambdaMeetProductsSpec.
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
(* What a meet answered                                                     *)
(* ======================================================================== *)

Lemma extract_value_answer_for :
  for_validity extract_value_answer_statement.
Proof. intros A r left right. destruct r; cbn; auto. Qed.

(* An answer of a meet of two parts is made of an answer of each. *)
Lemma combine_answer {A B C : Type} (rebuild : A -> B -> C)
    (res_a : meet_return_value A) (res_b : meet_return_value B)
    (left_a right_a : A) (left_b right_b : B) (answer : C) :
  meet_answer
    (combine_meet_return_values res_a res_b
       (fun _ =>
          rebuild (extract_value res_a left_a right_a)
            (extract_value res_b left_b right_b)))
    (rebuild left_a left_b) (rebuild right_a right_b) answer ->
  ∃ (a : A) (b : B),
    answer = rebuild a b
    ∧ meet_answer res_a left_a right_a a
    ∧ meet_answer res_b left_b right_b b.
Proof.
  destruct res_a, res_b; cbn; intros H; try destruct H as [ H | H ]; subst;
    eexists; eexists; (split; [ reflexivity | ]); split; cbn; auto.
Qed.

(* ======================================================================== *)
(* Three meets, one after the other                                         *)
(* ======================================================================== *)

(* [combine_results3] is [combine_results2] of the first meet and of
   [combine_results2] of the other two. *)
Lemma combine_results3_eq {A B C D : Type} (env : Meet_env.t)
    (meet_a : Meet_env.t -> A -> A -> option (Meet_result.t A))
    (meet_b : Meet_env.t -> B -> B -> option (Meet_result.t B))
    (meet_c : Meet_env.t -> C -> C -> option (Meet_result.t C))
    (left_a right_a : A) (left_b right_b : B) (left_c right_c : C)
    (rebuild : A -> B -> C -> D) :
  combine_results3 env meet_a meet_b meet_c left_a right_a left_b right_b
    left_c right_c rebuild
  = combine_results2 env meet_a
      (fun (env : Meet_env.t) (left right : B * C) =>
         combine_results2 env meet_b meet_c (fst left) (fst right) (snd left)
           (snd right) pair)
      left_a right_a (left_b, left_c) (right_b, right_c)
      (fun a bc => rebuild a (fst bc) (snd bc)).
Proof.
  unfold combine_results3, combine_results2.
  destruct (meet_a env left_a right_a) as [ [ r_a | res_a env1 ] | ];
    [ reflexivity | | reflexivity ].
  cbn.
  destruct (meet_b env1 left_b right_b) as [ [ r_b | res_b env2 ] | ];
    [ reflexivity | | reflexivity ].
  cbn.
  destruct (meet_c env2 left_c right_c) as [ [ r_c | res_c env3 ] | ];
    reflexivity.
Qed.

Lemma combine_results3_ok_for :
  for_validity combine_results3_ok_statement.
Proof.
  intros A B C D fits_a satisfied_a fits_b satisfied_b fits_c satisfied_c
    fits_d satisfied_d meet_a meet_b meet_c rebuild menv left_a right_a
    left_b right_b left_c right_c result tenv Hstable_a Hstable_b Hstable_c
    Hok_a Hok_b Hok_c Hfits_d Hsatisfied_d Henv Hla Hra Hlb Hrb Hlc Hrc Hmeet.
  rewrite combine_results3_eq in Hmeet.
  set (fits_bc :=
    fun (tenv : Typing_env.t) (bc : B * C) =>
      fits_b tenv (fst bc) ∧ fits_c tenv (snd bc)).
  set (satisfied_bc :=
    fun (statics : static_env) (valuation : env) (bc : B * C) =>
      satisfied_b statics valuation (fst bc)
      ∧ satisfied_c statics valuation (snd bc)).
  assert (Hstable_bc : part_stable fits_bc satisfied_bc).
  { destruct Hstable_b as [Hb1 Hb2], Hstable_c as [Hc1 Hc2]. split.
    - intros menv0 menv' bc Hextends [Hb Hc].
      split; [ exact (Hb1 _ _ _ Hextends Hb) | exact (Hc1 _ _ _ Hextends Hc) ].
    - intros statics tenv0 valuation valuation' bc [Hb Hc] Hextended
        [Hsb Hsc].
      split;
        [ exact (Hb2 _ _ _ _ _ Hb Hextended Hsb)
        | exact (Hc2 _ _ _ _ _ Hc Hextended Hsc) ]. }
  assert (Hok_bc :
    part_meet_ok fits_bc satisfied_bc
      (fun (env : Meet_env.t) (left right : B * C) =>
         combine_results2 env meet_b meet_c (fst left) (fst right) (snd left)
           (snd right) pair)).
  { intros menv0 [lb lc] [rb rc] result0 Henv0 [Hlb0 Hlc0] [Hrb0 Hrc0] Hmeet0.
    cbn [fst snd] in *.
    exact
      (combine_results2_ok_for B C (B * C)%type fits_b satisfied_b fits_c
         satisfied_c
         fits_bc satisfied_bc meet_b meet_c pair menv0 lb rb lc rc result0
         Hstable_b Hstable_c Hok_b Hok_c
         (fun tenv0 b c Hb Hc => conj Hb Hc)
         (fun statics valuation b c => iff_refl _) Henv0 Hlb0 Hrb0 Hlc0 Hrc0
         Hmeet0). }
  exact
    (combine_results2_ok_for A (B * C)%type D fits_a satisfied_a fits_bc
       satisfied_bc fits_d satisfied_d meet_a _
       (fun a bc => rebuild a (fst bc) (snd bc))
       menv left_a right_a (left_b, left_c) (right_b, right_c) result
       Hstable_a Hstable_bc Hok_a Hok_bc
       (fun tenv0 a bc Ha Hbc => Hfits_d tenv0 a _ _ Ha (proj1 Hbc) (proj2 Hbc))
       (fun statics valuation a bc => Hsatisfied_d statics valuation a _ _)
       Henv Hla Hra (conj Hlb Hlc) (conj Hrb Hrc) Hmeet).
Qed.

(* ======================================================================== *)
(* What [meet_mapping] answers, whatever the meet of the components         *)
(* ======================================================================== *)

Section Mapping_answer.

Context {K A : Type}.
Variable meet_data : Meet_env.t -> A -> A -> option (Meet_result.t A).
Variable get_left get_right : K -> option A.

Let step (state : mapping_state K A) (key : K) : mapping_state K A :=
  mapping_step meet_data key (get_left key) (get_right key) state.

(* A step that answers adds the key, and leaves "every component so far
   is the left one" true only if it was and the left input has the
   key; the same on the right. *)
Lemma mapping_step_flags (key : K) (acc acc' : mapping_acc K A) :
  step (Some (Or_bottom.Ok acc)) key = Some (Or_bottom.Ok acc') ->
  map fst (mapping acc') = key :: map fst (mapping acc)
  ∧ (all_left acc' = true -> all_left acc = true ∧ get_left key <> None)
  ∧ (all_right acc' = true -> all_right acc = true ∧ get_right key <> None).
Proof.
  unfold step, mapping_step.
  destruct (get_left key) as [ a | ], (get_right key) as [ b | ].
  - destruct (meet_data (mapping_env acc) a b) as [ [ r | res env ] | ];
      [ discriminate | | discriminate ].
    destruct res; cbv beta iota zeta; intros [= <-]; cbn;
      (split; [ reflexivity | ]); split; intros H;
      first [ discriminate H | split; [ exact H | discriminate ] ].
  - intros [= <-]. cbn. split; [ reflexivity | ]. split; intros H;
      first [ discriminate H | split; [ exact H | discriminate ] ].
  - intros [= <-]. cbn. split; [ reflexivity | ]. split; intros H;
      first [ discriminate H | split; [ exact H | discriminate ] ].
  - discriminate.
Qed.

Lemma mapping_fold_flags (todo : list K) :
  ∀ acc acc' : mapping_acc K A,
    List.fold_left step todo (Some (Or_bottom.Ok acc))
    = Some (Or_bottom.Ok acc') ->
    map fst (mapping acc') = rev todo ++ map fst (mapping acc)
    ∧ (all_left acc' = true ->
       all_left acc = true
       ∧ ∀ key : K, key ∈ todo -> get_left key <> None)
    ∧ (all_right acc' = true ->
       all_right acc = true
       ∧ ∀ key : K, key ∈ todo -> get_right key <> None).
Proof.
  induction todo as [ | key todo IH ]; intros acc acc' Hfold.
  - cbn in Hfold. injection Hfold as <-. split; [ reflexivity | ].
    split; intros H; (split; [ exact H | intros key [] ]).
  - cbn [List.fold_left] in Hfold.
    destruct (step (Some (Or_bottom.Ok acc)) key) as [ [ acc1 | ] | ]
      eqn:Hstep.
    + destruct (mapping_step_flags key acc acc1 Hstep)
        as (Hkeys1 & Hleft1 & Hright1).
      destruct (IH acc1 acc' Hfold) as (Hkeys & Hleft & Hright).
      split.
      { rewrite Hkeys, Hkeys1. cbn [rev]. rewrite <- app_assoc. reflexivity. }
      split; intros H.
      * destruct (Hleft H) as [H1 Hall]. destruct (Hleft1 H1) as [H0 Hkey].
        split; [ exact H0 | ].
        intros key' [ <- | Hin ]; [ exact Hkey | exact (Hall key' Hin) ].
      * destruct (Hright H) as [H1 Hall]. destruct (Hright1 H1) as [H0 Hkey].
        split; [ exact H0 | ].
        intros key' [ <- | Hin ]; [ exact Hkey | exact (Hall key' Hin) ].
    + rewrite mapping_fold_bottom in Hfold; [ discriminate Hfold | ].
      intros key'. reflexivity.
    + rewrite mapping_fold_none in Hfold; [ discriminate Hfold | ].
      intros key'. reflexivity.
Qed.

(* What [meet_mapping] answers: an input only if that input has every
   key, and otherwise what is made of one component for each key. *)
Lemma meet_mapping_answer {M : Type}
    (fold2 :
      (K -> option A -> option A -> mapping_state K A -> mapping_state K A) ->
      M -> M -> mapping_state K A -> mapping_state K A)
    (keys : list K) (env env' : Meet_env.t) (left right : M)
    (rebuild : list (K * A) -> M) (r : meet_return_value M) :
  (∀ (f :
        K -> option A -> option A -> mapping_state K A -> mapping_state K A)
     (init : mapping_state K A),
     fold2 f left right init
     = List.fold_left
         (fun state key => f key (get_left key) (get_right key) state)
         keys init) ->
  meet_mapping meet_data fold2 env left right rebuild
  = Some (Meet_result.Ok r env') ->
  match r with
  | Left_input => ∀ key : K, key ∈ keys -> get_left key <> None
  | Right_input => ∀ key : K, key ∈ keys -> get_right key <> None
  | Both_inputs =>
    (∀ key : K, key ∈ keys -> get_left key <> None)
    ∧ ∀ key : K, key ∈ keys -> get_right key <> None
  | New_result m =>
    ∃ l : list (K * A), m = rebuild l ∧ map fst l = rev keys
  end.
Proof.
  intros Hfold2 Hmeet. rewrite meet_mapping_unfold, Hfold2 in Hmeet.
  match type of Hmeet with
  | context [List.fold_left ?f keys ?init] =>
    destruct (List.fold_left f keys init) as [ [ res | ] | ] eqn:Hfold
  end; try discriminate Hmeet.
  destruct (mapping_fold_flags keys _ res Hfold) as (Hkeys & Hleft & Hright).
  cbn [mapping map] in Hkeys. rewrite app_nil_r in Hkeys.
  cbv zeta in Hmeet.
  destruct (all_left res), (all_right res); injection Hmeet as <- <-.
  - split; [ exact (proj2 (Hleft eq_refl)) | exact (proj2 (Hright eq_refl)) ].
  - exact (proj2 (Hleft eq_refl)).
  - exact (proj2 (Hright eq_refl)).
  - exists (mapping res). split; [ reflexivity | exact Hkeys ].
Qed.

End Mapping_answer.

(* ======================================================================== *)
(* Maps                                                                     *)
(* ======================================================================== *)

(* The map that [map_meet] makes of the components, the last first. *)
Lemma rebuild_map_rev {B : Type} (l acc : list B) :
  List.fold_left (fun m binding => binding :: m) l acc = rev l ++ acc.
Proof.
  revert acc. induction l as [ | b l IH ]; intros acc; [ reflexivity | ].
  cbn. rewrite IH, <- app_assoc. reflexivity.
Qed.

Lemma is_bound_of_lookup {K A : Type} (eqb : K -> K -> bool) (key : K)
    (bindings : Keyed.finite_map K A) :
  Keyed.lookup eqb key bindings <> None ->
  Keyed.is_bound eqb key bindings = true.
Proof.
  unfold Keyed.is_bound. destruct (Keyed.lookup eqb key bindings); congruence.
Qed.

Lemma map_meet_keys_for :
  for_validity map_meet_keys_statement.
Proof.
  intros K A eqb meet_data menv menv' left right answer r key Heqb Hmeet
    Hanswer.
  unfold map_meet in Hmeet.
  apply (meet_mapping_answer meet_data
           (fun key => Keyed.lookup eqb key left)
           (fun key => Keyed.lookup eqb key right) _
           (Keyed.keys_of_both eqb left right))
    in Hmeet; [ | intros f init; reflexivity ].
  assert (Hamong := keyed_among_keys_of_both K eqb Heqb A A key left right).
  assert (Hin :
    Keyed.is_bound eqb key left || Keyed.is_bound eqb key right = true ->
    key ∈ Keyed.keys_of_both eqb left right).
  { intros Hbound. rewrite <- Hamong in Hbound.
    exact (proj1 (keyed_among_in K eqb Heqb key _) Hbound). }
  assert (Hleft :
    (∀ key : K,
       key ∈ Keyed.keys_of_both eqb left right ->
       Keyed.lookup eqb key left <> None) ->
    Keyed.is_bound eqb key left
    = Keyed.is_bound eqb key left || Keyed.is_bound eqb key right).
  { intros Hall. destruct (Keyed.is_bound eqb key right) eqn:Hright;
      [ | rewrite orb_false_r; reflexivity ].
    rewrite orb_true_r. apply is_bound_of_lookup, Hall, Hin.
    rewrite ?Hright. apply orb_true_r. }
  assert (Hright :
    (∀ key : K,
       key ∈ Keyed.keys_of_both eqb left right ->
       Keyed.lookup eqb key right <> None) ->
    Keyed.is_bound eqb key right
    = Keyed.is_bound eqb key left || Keyed.is_bound eqb key right).
  { intros Hall. destruct (Keyed.is_bound eqb key left) eqn:Hl;
      [ | reflexivity ].
    cbn [orb]. apply is_bound_of_lookup, Hall, Hin. rewrite ?Hl. reflexivity. }
  destruct r as [ | | | m ]; cbn [meet_answer] in Hanswer.
  - subst answer. exact (Hleft Hmeet).
  - subst answer. exact (Hright Hmeet).
  - destruct Hanswer as [ -> | -> ];
      [ exact (Hleft (proj1 Hmeet)) | exact (Hright (proj2 Hmeet)) ].
  - subst answer. destruct Hmeet as (l & -> & Hkeys).
    rewrite rebuild_map_rev, app_nil_r, keyed_is_bound_among, map_rev, Hkeys,
      rev_involutive.
    exact Hamong.
Qed.

Section Map_stable.

Context {K A : Type}.
Variable eqb : K -> K -> bool.
Variable fits : Typing_env.t -> A -> Prop.
Variable satisfied : K -> static_env -> env -> A -> Prop.
Hypothesis Hstable : ∀ key : K, part_stable fits (satisfied key).

Lemma map_part_stable :
  part_stable (map_fits eqb fits) (map_satisfied eqb satisfied).
Proof.
  split.
  - intros menv menv' bindings Hextends [Hdistinct Hfits].
    split; [ exact Hdistinct | ]. intros key a Hlookup.
    exact (proj1 (Hstable key) menv menv' a Hextends (Hfits key a Hlookup)).
  - intros statics tenv valuation valuation' bindings [_ Hfits] Hextended
      Hsatisfied key a Hlookup.
    exact (proj2 (Hstable key) statics tenv valuation valuation' a
             (Hfits key a Hlookup) Hextended (Hsatisfied key a Hlookup)).
Qed.

End Map_stable.

Lemma map_meet_ok_for :
  for_validity map_meet_ok_statement.
Proof.
  intros K A eqb fits satisfied meet_data Heqb Hstable Hok menv left right
    result Henv Hleft Hright Hmeet.
  destruct (map_part_stable eqb fits satisfied Hstable)
    as [Hstable1 Hstable2].
  unfold map_meet in Hmeet.
  destruct
    (meet_mapping_ok_for K A (Keyed.finite_map K A) fits satisfied meet_data _
       (Keyed.keys_of_both eqb left right)
       (fun key => Keyed.lookup eqb key left)
       (fun key => Keyed.lookup eqb key right) menv left right _ result
       (fun f init => eq_refl) Hstable Hok Henv (proj2 Hleft) (proj2 Hright)
       Hmeet)
    as [Hkeeps Hsound].
  assert (Hrebuilt :
    ∀ (tenv' : Typing_env.t) (l : list (K * A)),
      map fst l = rev (Keyed.keys_of_both eqb left right) ->
      (∀ (key : K) (a : A), (key, a) ∈ l -> fits tenv' a) ->
      map_fits eqb fits tenv'
        (List.fold_left (fun m binding => binding :: m) l [])).
  { intros tenv' l Hkeys Hfits. rewrite rebuild_map_rev, app_nil_r. split.
    - unfold keys. rewrite map_rev, Hkeys, rev_involutive.
      apply keyed_keys_of_both_distinct, Heqb.
    - intros key a Hlookup. apply (keyed_lookup_in eqb Heqb) in Hlookup.
      rewrite <- in_rev in Hlookup. exact (Hfits key a Hlookup). }
  split.
  - intros r menv' ->.
    destruct (Hkeeps r menv' eq_refl) as (Hextends & Henv' & Hnames & Hnew).
    split; [ exact Hextends | ]. split; [ exact Henv' | ].
    split; [ | exact Hnames ].
    destruct r as [ | | | m ]; cbn [extract_value].
    + exact (Hstable1 menv menv' left Hextends Hleft).
    + exact (Hstable1 menv menv' right Hextends Hright).
    + exact (Hstable1 menv menv' left Hextends Hleft).
    + destruct (Hnew m eq_refl) as (l & -> & Hkeys & Hfits).
      exact (Hrebuilt _ l Hkeys Hfits).
  - intros statics valuation Hstatics Hvaluation Hs1 Hs2.
    destruct (Hsound statics valuation Hstatics Hvaluation Hs1 Hs2)
      as (r & menv' & valuation' & -> & Hextended & Hvaluation' & Hvalued
          & Hnew).
    exists r, menv', valuation'. split; [ reflexivity | ].
    split; [ exact Hextended | ]. split; [ exact Hvaluation' | ].
    split; [ | exact Hvalued ].
    destruct r as [ | | | m ]; cbn [extract_value].
    + exact (Hstable2 statics _ valuation valuation' left Hleft Hextended Hs1).
    + exact (Hstable2 statics _ valuation valuation' right Hright Hextended
               Hs2).
    + exact (Hstable2 statics _ valuation valuation' left Hleft Hextended Hs1).
    + destruct (Hnew m eq_refl) as (l & -> & Hkeys & Hall).
      intros key a Hlookup. rewrite rebuild_map_rev, app_nil_r in Hlookup.
      apply (keyed_lookup_in eqb Heqb) in Hlookup.
      rewrite <- in_rev in Hlookup. exact (proj2 (Hall key a Hlookup)).
Qed.

(* ======================================================================== *)
(* The meet of two types of a position that the value has                   *)
(* ======================================================================== *)

(* The type of a position is true: the value has something there,
   [value_at], which has the type. *)
Definition present_satisfied (value_at : static_env -> option value)
    (statics : static_env) (valuation : env) (ty : flambda_type) : Prop :=
  ∃ w : value,
    value_at statics = Some w ∧ type_satisfied statics valuation w ty.

Lemma type_fits_extends (menv menv' : Meet_env.t) (ty : flambda_type) :
  meet_env_extends menv menv' ->
  type_fits (Meet_env.typing_env menv) ty ->
  type_fits (Meet_env.typing_env menv') ty.
Proof.
  intros Hextends.
  exact (tenv_extends_type_fits _ _ ty (meet_env_extends_tenv _ _ Hextends)).
Qed.

Lemma present_part_stable (value_at : static_env -> option value) :
  part_stable type_fits (present_satisfied value_at).
Proof.
  split.
  - intros menv menv' ty. apply type_fits_extends.
  - intros statics tenv valuation valuation' ty Hfits Hextended
      (w & Hw & Hsatisfied).
    exists w. split; [ exact Hw | ].
    exact (type_satisfied_extended statics tenv valuation valuation' w ty
             Hfits Hextended Hsatisfied).
Qed.

Section Present.

Variable mh : meet_expanded_head.
Hypothesis Hkeeps : meet_expanded_head_keeps mh.
Hypothesis Hsound : meet_expanded_head_sound mh.

Lemma meet_type_present_ok (value_at : static_env -> option value) :
  part_meet_ok type_fits (present_satisfied value_at) (meet mh).
Proof.
  intros menv t1 t2 result Henv H1 H2 Hmeet. split.
  - exact (proj1 (meet_type_ok_for mh (Value_const Null) Hkeeps Hsound menv t1
                    t2
                    result Henv H1 H2 Hmeet)).
  - intros statics valuation Hstatics Hvaluation (w1 & Hw1 & Hs1)
      (w2 & Hw2 & Hs2).
    assert (Hsame : w2 = w1) by congruence. subst w2.
    destruct (proj2 (meet_type_ok_for mh w1 Hkeeps Hsound menv t1 t2 result Henv
                       H1 H2 Hmeet)
                statics valuation Hstatics Hvaluation Hs1 Hs2)
      as (r & menv' & valuation' & Hresult & Hextended & Hvaluation'
          & Hanswer & Hvalued).
    exists r, menv', valuation'. split; [ exact Hresult | ].
    split; [ exact Hextended | ]. split; [ exact Hvaluation' | ].
    split; [ | exact Hvalued ]. exists w1. split; [ exact Hw1 | exact Hanswer ].
Qed.

End Present.

(* ======================================================================== *)
(* The fields of a block                                                    *)
(* ======================================================================== *)

Lemma product_satisfied_fields_for :
  for_validity product_satisfied_fields_statement.
Proof.
  intros fields statics valuation tys. unfold product_satisfied.
  rewrite (fields_satisfied_nth (type_satisfied statics valuation) tys fields).
  split.
  - intros Hall. split.
    + destruct (Nat.le_gt_cases (length tys) (length fields)) as [ H | H ];
        [ exact H | exfalso ].
      destruct (nth_error tys (length fields)) as [ ty | ] eqn:Hty.
      * destruct (Hall _ ty Hty) as (field & Hfield & _).
        rewrite (proj2 (nth_error_None fields (length fields))
                   (Nat.le_refl _))
          in Hfield.
        discriminate Hfield.
      * apply nth_error_None in Hty. lia.
    + intros i ty field Hty Hfield.
      destruct (Hall i ty Hty) as (field' & Hfield' & Hsatisfied).
      congruence.
  - intros [Hlength Hall] i ty Hty.
    destruct (nth_error fields i) as [ field | ] eqn:Hfield.
    + exists field. split; [ reflexivity | exact (Hall i ty field Hty Hfield) ].
    + exfalso. apply nth_error_None in Hfield.
      assert (Hlt : i < length tys).
      { apply nth_error_Some. rewrite Hty. discriminate. }
      lia.
Qed.

(* The list that [meet_array_of_types] makes of the components. *)
Definition array_rebuild (length : nat) (l : list (nat * flambda_type))
    : list flambda_type :=
  match l with
  | [] => []
  | (_key, data) :: _ =>
    map
      (fun i =>
         match Keyed.lookup Nat.eqb i l with
         | Some data => data
         | None => data
         end)
      (seq 0 length)
  end.

Lemma meet_int_indexed_product_unfold (mh : meet_expanded_head)
    (menv : Meet_env.t) (fields1 fields2 : list flambda_type) :
  meet_int_indexed_product mh menv fields1 fields2
  = meet_mapping (meet mh)
      (fun f left right init =>
         List.fold_left
           (fun acc i => f i (nth_error left i) (nth_error right i) acc)
           (seq 0 (Nat.max (length fields1) (length fields2)))
           init)
      menv fields1 fields2
      (array_rebuild (Nat.max (length fields1) (length fields2))).
Proof. reflexivity. Qed.

Lemma nth_error_seq (n : nat) :
  ∀ start i : nat,
    nth_error (seq start n) i = if i <? n then Some (start + i) else None.
Proof.
  induction n as [ | n IH ]; intros start i.
  - destruct i; reflexivity.
  - destruct i as [ | i ]; cbn [seq nth_error].
    + rewrite Nat.add_0_r. reflexivity.
    + rewrite IH. change (S i <? S n) with (i <? n).
      destruct (i <? n); [ | reflexivity ]. f_equal. lia.
Qed.

Lemma nat_eqb_decides_equality : decides_equality Nat.eqb.
Proof. exact Nat.eqb_eq. Qed.

Section Array_rebuild.

Variable n : nat.
Variable l : list (nat * flambda_type).
Hypothesis Hkeys : map fst l = rev (seq 0 n).

Lemma array_rebuild_length : length (array_rebuild n l) = n.
Proof.
  unfold array_rebuild. destruct l as [ | [key data] rest ].
  - cbn in Hkeys. symmetry in Hkeys. apply (f_equal (@length nat)) in Hkeys.
    rewrite length_rev, length_seq in Hkeys. cbn in Hkeys. cbn. lia.
  - rewrite length_map, length_seq. reflexivity.
Qed.

Lemma array_rebuild_nth (i : nat) (ty : flambda_type) :
  nth_error (array_rebuild n l) i = Some ty -> (i, ty) ∈ l.
Proof.
  unfold array_rebuild. destruct l as [ | [key data] rest ] eqn:Hl.
  - destruct i; discriminate.
  - rewrite <- Hl in *. rewrite nth_error_map, nth_error_seq.
    destruct (i <? n) eqn:Hlt; [ | discriminate ].
    cbn [option_map Nat.add].
    destruct (Keyed.lookup Nat.eqb i l) as [ found | ] eqn:Hlookup.
    + intros [= <-].
      exact (keyed_lookup_in Nat.eqb nat_eqb_decides_equality i found l
               Hlookup).
    + exfalso.
      apply (keyed_lookup_none_not_in Nat.eqb nat_eqb_decides_equality)
        in Hlookup.
      apply Hlookup. rewrite Hkeys, <- in_rev. apply in_seq.
      apply Nat.ltb_lt in Hlt. lia.
Qed.

End Array_rebuild.

Lemma meet_int_indexed_product_ok_for :
  for_validity meet_int_indexed_product_ok_statement.
Proof.
  intros mh fields Hkeeps Hsound menv fields1 fields2 result Henv Hfits1
    Hfits2 Hmeet.
  rewrite meet_int_indexed_product_unfold in Hmeet.
  set (n := Nat.max (length fields1) (length fields2)) in Hmeet.
  set (satisfied :=
    fun i : nat => present_satisfied (fun _ => nth_error fields i)).
  destruct
    (meet_mapping_ok_for nat flambda_type (list flambda_type) type_fits
       satisfied
       (meet mh) _ (seq 0 n) (nth_error fields1) (nth_error fields2) menv
       fields1 fields2 _ result (fun f init => eq_refl)
       (fun i => present_part_stable _)
       (fun i => meet_type_present_ok mh Hkeeps Hsound _) Henv
       (fun i ty H => Hfits1 ty (nth_error_In _ _ H))
       (fun i ty H => Hfits2 ty (nth_error_In _ _ H)) Hmeet)
    as [Hkeeps' Hsound'].
  split.
  - intros r menv' ->.
    destruct (Hkeeps' r menv' eq_refl) as (Hextends & Henv' & Hnames & Hnew).
    split; [ exact Hextends | ]. split; [ exact Henv' | ].
    split; [ | exact Hnames ].
    destruct r as [ | | | m ]; cbn [extract_value].
    + intros ty Hin. exact (type_fits_extends _ _ ty Hextends (Hfits1 ty Hin)).
    + intros ty Hin. exact (type_fits_extends _ _ ty Hextends (Hfits2 ty Hin)).
    + intros ty Hin. exact (type_fits_extends _ _ ty Hextends (Hfits1 ty Hin)).
    + destruct (Hnew m eq_refl) as (l & -> & Hkeys & Hfits).
      intros ty Hin. apply In_nth_error in Hin as [i Hi].
      exact (Hfits i ty (array_rebuild_nth n l Hkeys i ty Hi)).
  - intros statics valuation Hstatics Hvaluation Hs1 Hs2.
    destruct (Hsound' statics valuation Hstatics Hvaluation Hs1 Hs2)
      as (r & menv' & valuation' & -> & Hextended & Hvaluation' & Hvalued
          & Hnew).
    exists r, menv', valuation'. split; [ reflexivity | ].
    split; [ exact Hextended | ]. split; [ exact Hvaluation' | ].
    split; [ | exact Hvalued ].
    assert (Hinput :
      ∀ fs : list flambda_type,
        product_fits (Meet_env.typing_env menv) fs ->
        product_satisfied fields statics valuation fs ->
        product_satisfied fields statics valuation' fs).
    { intros fs Hfits Hs i ty Hnth.
      destruct (Hs i ty Hnth) as (field & Hfield & Hty).
      exists field. split; [ exact Hfield | ].
      exact (type_satisfied_extended statics _ valuation valuation' field ty
               (Hfits ty (nth_error_In _ _ Hnth)) Hextended Hty). }
    destruct r as [ | | | m ]; cbn [extract_value].
    + exact (Hinput fields1 Hfits1 Hs1).
    + exact (Hinput fields2 Hfits2 Hs2).
    + exact (Hinput fields1 Hfits1 Hs1).
    + destruct (Hnew m eq_refl) as (l & -> & Hkeys & Hall).
      intros i ty Hnth.
      exact (proj2 (Hall i ty (array_rebuild_nth n l Hkeys i ty Hnth))).
Qed.

Lemma meet_int_indexed_product_length_for :
  for_validity meet_int_indexed_product_length_statement.
Proof.
  intros mh menv menv' fields1 fields2 answer r Hmeet Hanswer.
  rewrite meet_int_indexed_product_unfold in Hmeet.
  set (n := Nat.max (length fields1) (length fields2)) in *.
  apply (meet_mapping_answer (meet mh) (nth_error fields1)
           (nth_error fields2) _ (seq 0 n))
    in Hmeet; [ | intros f init; reflexivity ].
  assert (Hlength :
    ∀ fs : list flambda_type,
      (∀ i : nat, i ∈ seq 0 n -> nth_error fs i <> None) -> n <= length fs).
  { intros fs Hall.
    destruct (Nat.le_gt_cases n (length fs)) as [ H | H ];
      [ exact H | exfalso ].
    apply (Hall (length fs)); [ apply in_seq; lia | ].
    apply nth_error_None, Nat.le_refl. }
  destruct r as [ | | | m ]; cbn [meet_answer] in Hanswer.
  - subst answer. specialize (Hlength fields1 Hmeet). unfold n in *. lia.
  - subst answer. specialize (Hlength fields2 Hmeet). unfold n in *. lia.
  - assert (H1 := Hlength fields1 (proj1 Hmeet)).
    assert (H2 := Hlength fields2 (proj2 Hmeet)).
    destruct Hanswer as [ -> | -> ]; unfold n in *; lia.
  - subst answer. destruct Hmeet as (l & -> & Hkeys).
    exact (array_rebuild_length n l Hkeys).
Qed.

(* ======================================================================== *)
(* Function types                                                           *)
(* ======================================================================== *)

Lemma meet_function_type_ok_for :
  for_validity meet_function_type_ok_statement.
Proof.
  intros functions slot menv ft1 ft2 result Henv _ _ Hmeet.
  unfold meet_function_type in Hmeet.
  destruct ft1 as [ ft1 | ], ft2 as [ ft2 | ].
  - injection Hmeet as <-. unfold meet_code_id.
    destruct (Pos.eqb (function_type_code_id ft1) (function_type_code_id ft2))
      eqn:Hcode.
    + cbn [map_result].
      apply (meet_call_ok_same_env _ _ menv _ _ Both_inputs Henv I).
      intros statics valuation _ H1 _. exact H1.
    + cbn [map_result]. apply meet_call_ok_bottom.
      intros statics valuation _ _ (code1 & Hcode1 & H1)
        (code2 & Hcode2 & H2).
      rewrite Hcode1 in Hcode2. injection Hcode2 as <-. subst code1.
      rewrite H2, Pos.eqb_refl in Hcode. discriminate Hcode.
  - injection Hmeet as <-.
    apply (meet_call_ok_same_env _ _ menv _ _ Left_input Henv I).
    intros statics valuation _ H1 _. exact H1.
  - injection Hmeet as <-.
    apply (meet_call_ok_same_env _ _ menv _ _ Right_input Henv I).
    intros statics valuation _ _ H2. exact H2.
  - injection Hmeet as <-.
    apply (meet_call_ok_same_env _ _ menv _ _ Both_inputs Henv I).
    intros statics valuation _ H1 _. exact H1.
Qed.

(* ======================================================================== *)
(* Entries of closures                                                      *)
(* ======================================================================== *)

(* A meet that is right for a way of fitting is right for a stronger
   one that its answers have. *)
Lemma meet_call_ok_fits {A : Type} (fits fits' : Typing_env.t -> A -> Prop)
    (satisfied : static_env -> env -> A -> Prop) (menv : Meet_env.t)
    (a1 a2 : A) (result : Meet_result.t A) :
  meet_call_ok fits satisfied menv a1 a2 result ->
  (∀ (r : meet_return_value A) (menv' : Meet_env.t),
     result = Meet_result.Ok r menv' ->
     fits (Meet_env.typing_env menv') (extract_value r a1 a2) ->
     fits' (Meet_env.typing_env menv') (extract_value r a1 a2)) ->
  meet_call_ok fits' satisfied menv a1 a2 result.
Proof.
  intros [Hkeeps Hsound] Hstronger. split; [ | exact Hsound ].
  intros r menv' Hresult.
  destruct (Hkeeps r menv' Hresult) as (Hextends & Henv' & Hanswer & Hnames).
  split; [ exact Hextends | ]. split; [ exact Henv' | ].
  split; [ exact (Hstronger r menv' Hresult Hanswer) | exact Hnames ].
Qed.

(* The type of a closure of a set fits: it does as a type, and it is
   of kind value. *)
Definition closure_type_fits (tenv : Typing_env.t) (ty : flambda_type)
    : Prop :=
  type_fits tenv ty ∧ kind ty = Value.

Lemma closure_type_part_stable (value_at : static_env -> option value) :
  part_stable closure_type_fits (present_satisfied value_at).
Proof.
  destruct (present_part_stable value_at) as [Hstable1 Hstable2]. split.
  - intros menv menv' ty Hextends [Hfits Hkind].
    split; [ exact (Hstable1 menv menv' ty Hextends Hfits) | exact Hkind ].
  - intros statics tenv valuation valuation' ty [Hfits _].
    exact (Hstable2 statics tenv valuation valuation' ty Hfits).
Qed.

Lemma meet_closure_type_ok (mh : meet_expanded_head)
    (value_at : static_env -> option value) :
  meet_expanded_head_keeps mh ->
  meet_expanded_head_sound mh ->
  part_meet_ok closure_type_fits (present_satisfied value_at) (meet mh).
Proof.
  intros Hkeeps Hsound menv t1 t2 result Henv [H1 Hkind1] [H2 _] Hmeet.
  apply (meet_call_ok_fits type_fits).
  - exact (meet_type_present_ok mh Hkeeps Hsound value_at menv t1 t2 result
             Henv H1 H2 Hmeet).
  - intros r menv' -> Hanswer. split; [ exact Hanswer | ].
    rewrite (proj2 (proj2 (meet_type_keeps_for mh menv menv' t1 t2 r Hkeeps Henv
                             H1 H2 Hmeet))).
    exact Hkind1.
Qed.

(* In a map that binds a key once, a binding of the list is what
   [lookup] finds. *)
Lemma in_lookup_distinct {K A : Type} (eqb : K -> K -> bool)
    (Heqb : decides_equality eqb) (bindings : list (K * A)) :
  Keyed.distinct eqb (map fst bindings) = true ->
  ∀ (key : K) (a : A),
    (key, a) ∈ bindings -> Keyed.lookup eqb key bindings = Some a.
Proof.
  induction bindings as [ | [bound b] rest IH ]; intros Hdistinct key a Hin;
    [ destruct Hin | ].
  cbn in Hdistinct. apply andb_true_iff in Hdistinct as [Hfresh Hdistinct].
  rewrite keyed_lookup_cons. destruct Hin as [ [= <- <-] | Hin ].
  - rewrite (keyed_eqb_refl eqb Heqb). reflexivity.
  - destruct (eqb key bound) eqn:Hkey; [ | exact (IH Hdistinct key a Hin) ].
    exfalso. apply Heqb in Hkey. subst bound.
    apply negb_true_iff in Hfresh.
    assert (Hamong : Keyed.among eqb key (map fst rest) = true).
    { apply (keyed_among_in K eqb Heqb). apply in_map_iff.
      exists (key, a). split; [ reflexivity | exact Hin ]. }
    congruence.
Qed.

(* An entry fits when its three maps do. *)
Lemma closures_entry_fits_maps (tenv : Typing_env.t)
    (function_types : finite_map (Or_unknown.t function_type))
    (closure_types value_slot_types : finite_map flambda_type) :
  closures_entry_fits tenv
    (Closures_entry function_types closure_types value_slot_types)
  <-> map_fits Pos.eqb
        (fun (_ : Typing_env.t) (_ : Or_unknown.t function_type) => True)
        tenv function_types
      ∧ map_fits Pos.eqb closure_type_fits tenv closure_types
      ∧ map_fits Pos.eqb type_fits tenv value_slot_types.
Proof.
  unfold closures_entry_fits, map_fits. split.
  - intros ((index & Hwf) & Hvars & Haliases & Hequations).
    apply closures_entry_well_formed_iff in Hwf
      as (_ & (Hd1 & Hd2 & Hd3) & Hclosures & Hvalue_slots).
    split; [ split; [ exact Hd1 | intros key a _; exact I ] | ].
    split; (split; [ assumption | ]).
    + intros g ty Hlookup.
      destruct (Hclosures g ty Hlookup) as [Hkind Hwf].
      split; [ | exact Hkind ]. split; [ exact Hwf | ].
      assert (Hin := lookup_in g ty closure_types Hlookup).
      split; [ | split ].
      * intros x Hx. apply Hvars. cbn. apply in_or_app. left.
        apply in_flat_map. exists (g, ty). split; [ exact Hin | exact Hx ].
      * intros s kind Hs. apply Haliases. cbn. apply in_or_app. left.
        apply in_flat_map. exists (g, ty). split; [ exact Hin | exact Hs ].
      * intros n kind Hn. apply Hequations. cbn. apply in_or_app. left.
        apply in_flat_map. exists (g, ty). split; [ exact Hin | exact Hn ].
    + intros w ty Hlookup. split; [ exact (Hvalue_slots w ty Hlookup) | ].
      assert (Hin := lookup_in w ty value_slot_types Hlookup).
      split; [ | split ].
      * intros x Hx. apply Hvars. cbn. apply in_or_app. right.
        apply in_flat_map. exists (w, ty). split; [ exact Hin | exact Hx ].
      * intros s kind Hs. apply Haliases. cbn. apply in_or_app. right.
        apply in_flat_map. exists (w, ty). split; [ exact Hin | exact Hs ].
      * intros n kind Hn. apply Hequations. cbn. apply in_or_app. right.
        apply in_flat_map. exists (w, ty). split; [ exact Hin | exact Hn ].
  - intros ((Hd1 & _) & (Hd2 & Hclosures) & (Hd3 & Hvalue_slots)).
    split; [ | split; [ | split ] ].
    + exists
        (Row_like_index
           (Row_like_index_domain.At_least
              (Set_of_closures_contents
                 (domain function_types ++ domain closure_types)
                 (domain value_slot_types)))
           tt).
      apply closures_entry_well_formed_iff.
      split; [ | split; [ auto | split ] ].
      * unfold closures_entry_slots_fit. cbn. rewrite !andb_true_iff.
        split; [ split; [ split | ] | reflexivity ];
          apply all_among; intros name Hname; rewrite ?among_app, ?Hname;
          auto using orb_true_r.
      * intros g ty Hlookup. destruct (Hclosures g ty Hlookup) as [Hty Hkind].
        split; [ exact Hkind | exact (proj1 Hty) ].
      * intros w ty Hlookup. exact (proj1 (Hvalue_slots w ty Hlookup)).
    + intros x Hx. cbn in Hx.
      apply in_app_or in Hx as [ Hx | Hx ];
        apply in_flat_map in Hx as ([key ty] & Hin & Hx); cbn [snd] in Hx.
      * apply (in_lookup_distinct Pos.eqb pos_eqb_decides_equality _ Hd2)
          in Hin.
        exact (proj1 (proj2 (proj1 (Hclosures key ty Hin))) x Hx).
      * apply (in_lookup_distinct Pos.eqb pos_eqb_decides_equality _ Hd3)
          in Hin.
        exact (proj1 (proj2 (Hvalue_slots key ty Hin)) x Hx).
    + intros s kind Hs. cbn in Hs.
      apply in_app_or in Hs as [ Hs | Hs ];
        apply in_flat_map in Hs as ([key ty] & Hin & Hs); cbn [snd] in Hs.
      * apply (in_lookup_distinct Pos.eqb pos_eqb_decides_equality _ Hd2)
          in Hin.
        exact (proj1 (proj2 (proj2 (proj1 (Hclosures key ty Hin)))) s kind Hs).
      * apply (in_lookup_distinct Pos.eqb pos_eqb_decides_equality _ Hd3)
          in Hin.
        exact (proj1 (proj2 (proj2 (Hvalue_slots key ty Hin))) s kind Hs).
    + intros n kind Hn. cbn in Hn.
      apply in_app_or in Hn as [ Hn | Hn ];
        apply in_flat_map in Hn as ([key ty] & Hin & Hn); cbn [snd] in Hn.
      * apply (in_lookup_distinct Pos.eqb pos_eqb_decides_equality _ Hd2)
          in Hin.
        exact (proj2 (proj2 (proj2 (proj1 (Hclosures key ty Hin)))) n kind Hn).
      * apply (in_lookup_distinct Pos.eqb pos_eqb_decides_equality _ Hd3)
          in Hin.
        exact (proj2 (proj2 (proj2 (Hvalue_slots key ty Hin))) n kind Hn).
Qed.

(* An entry is true of a closure when its three maps are. *)
Lemma closures_entry_holds_maps (v : value) (slot : function_slot)
    (functions : list (function_slot * code_id))
    (value_slots : list (value_slot * value)) (statics : static_env)
    (valuation : env)
    (function_types : finite_map (Or_unknown.t function_type))
    (closure_types value_slot_types : finite_map flambda_type) :
  closures_entry_holds v slot functions value_slots statics valuation
    (Closures_entry function_types closure_types value_slot_types)
  <-> map_satisfied Pos.eqb (function_type_satisfied functions) statics
        valuation function_types
      ∧ map_satisfied Pos.eqb
          (fun g =>
             present_satisfied
               (fun statics => project_function_slot statics slot g v))
          statics valuation closure_types
      ∧ map_satisfied Pos.eqb
          (fun w => present_satisfied (fun _ => lookup w value_slots))
          statics valuation value_slot_types.
Proof.
  unfold closures_entry_holds, closures_entry_slots_present, map_satisfied,
    present_satisfied, function_type_satisfied.
  rewrite closures_entry_lookup_iff.
  split.
  - intros ((Hfunctions & Hclosures & Hvalue_slots)
            & (Pfunctions & Pclosures & Pvalue_slots)).
    split; [ | split ].
    + intros g function_type Hlookup.
      assert (Hbound : is_bound g functions = true).
      { apply Pfunctions. unfold is_bound.
        change (lookup g function_types)
          with (Keyed.lookup Pos.eqb g function_types).
        rewrite Hlookup. reflexivity. }
      unfold is_bound in Hbound.
      destruct (lookup g functions) as [ code | ] eqn:Hcode;
        [ | discriminate Hbound ].
      exists code. split; [ reflexivity | ].
      destruct function_type as [ function_type | ]; [ | exact I ].
      exact (Hfunctions g function_type code Hlookup Hcode).
    + intros g ty Hlookup.
      destruct (Pclosures g) as [closure Hclosure].
      { unfold is_bound.
        change (lookup g closure_types)
          with (Keyed.lookup Pos.eqb g closure_types).
        rewrite Hlookup. reflexivity. }
      exists closure. split; [ exact Hclosure | ].
      exact (Hclosures g ty closure Hlookup Hclosure).
    + intros w ty Hlookup.
      assert (Hbound : is_bound w value_slots = true).
      { apply Pvalue_slots. unfold is_bound.
        change (lookup w value_slot_types)
          with (Keyed.lookup Pos.eqb w value_slot_types).
        rewrite Hlookup. reflexivity. }
      unfold is_bound in Hbound.
      destruct (lookup w value_slots) as [ contents | ] eqn:Hcontents;
        [ | discriminate Hbound ].
      exists contents. split; [ reflexivity | ].
      exact (Hvalue_slots w ty contents Hlookup Hcontents).
  - intros (Hfunctions & Hclosures & Hvalue_slots). split; [ split | split ].
    + intros g function_type code Hlookup Hcode.
      destruct (Hfunctions g _ Hlookup) as (code' & Hcode' & Hsame).
      congruence.
    + split.
      * intros g ty closure Hlookup Hclosure.
        destruct (Hclosures g ty Hlookup) as (w & Hw & Hsatisfied).
        congruence.
      * intros w ty contents Hlookup Hcontents.
        destruct (Hvalue_slots w ty Hlookup) as (w' & Hw' & Hsatisfied).
        congruence.
    + intros g Hbound. unfold is_bound in Hbound.
      destruct (lookup g function_types) as [ function_type | ] eqn:Hlookup;
        [ | discriminate Hbound ].
      destruct (Hfunctions g _ Hlookup) as (code & Hcode & _).
      unfold is_bound. rewrite Hcode. reflexivity.
    + split.
      * intros g Hbound. unfold is_bound in Hbound.
        destruct (lookup g closure_types) as [ ty | ] eqn:Hlookup;
          [ | discriminate Hbound ].
        destruct (Hclosures g ty Hlookup) as (closure & Hclosure & _).
        exists closure. exact Hclosure.
      * intros w Hbound. unfold is_bound in Hbound.
        destruct (lookup w value_slot_types) as [ ty | ] eqn:Hlookup;
          [ | discriminate Hbound ].
        destruct (Hvalue_slots w ty Hlookup) as (contents & Hcontents & _).
        unfold is_bound. rewrite Hcontents. reflexivity.
Qed.

Lemma function_type_part_stable (functions : list (function_slot * code_id))
    (g : function_slot) :
  part_stable
    (fun (_ : Typing_env.t) (_ : Or_unknown.t function_type) => True)
    (function_type_satisfied functions g).
Proof.
  split.
  - intros menv menv' function_type _ _. exact I.
  - intros statics tenv valuation valuation' function_type _ _ Hsatisfied.
    exact Hsatisfied.
Qed.

Lemma meet_closures_entry_ok_for :
  for_validity meet_closures_entry_ok_statement.
Proof.
  intros mh v slot functions value_slots Hkeeps Hsound menv entry1 entry2
    result Henv Hfits1 Hfits2 Hmeet.
  destruct entry1 as [function_types1 closure_types1 value_slot_types1].
  destruct entry2 as [function_types2 closure_types2 value_slot_types2].
  apply closures_entry_fits_maps in Hfits1 as (Hf1 & Hc1 & Hv1).
  apply closures_entry_fits_maps in Hfits2 as (Hf2 & Hc2 & Hv2).
  cbn [meet_closures_entry] in Hmeet.
  set (closure_at :=
    fun (g : function_slot) (statics : static_env) =>
      project_function_slot statics slot g v).
  set (value_slot_at :=
    fun (w : value_slot) (_ : static_env) => lookup w value_slots).
  exact
    (combine_results3_ok_for _ _ _ _
       (map_fits Pos.eqb (fun _ _ => True))
       (map_satisfied Pos.eqb (function_type_satisfied functions))
       (map_fits Pos.eqb closure_type_fits)
       (map_satisfied Pos.eqb (fun g => present_satisfied (closure_at g)))
       (map_fits Pos.eqb type_fits)
       (map_satisfied Pos.eqb (fun w => present_satisfied (value_slot_at w)))
       closures_entry_fits
       (closures_entry_holds v slot functions value_slots)
       (map_meet Pos.eqb meet_function_type)
       (meet_product_function_slot_indexed mh)
       (meet_product_value_slot_indexed mh) Closures_entry menv
       function_types1 function_types2 closure_types1 closure_types2
       value_slot_types1 value_slot_types2 result
       (map_part_stable Pos.eqb _ _ (function_type_part_stable functions))
       (map_part_stable Pos.eqb _ _
          (fun g => closure_type_part_stable (closure_at g)))
       (map_part_stable Pos.eqb _ _
          (fun w => present_part_stable (value_slot_at w)))
       (map_meet_ok_for _ _ Pos.eqb _ _ meet_function_type
          pos_eqb_decides_equality (function_type_part_stable functions)
          (meet_function_type_ok_for functions))
       (map_meet_ok_for _ _ Pos.eqb _ _ (meet mh) pos_eqb_decides_equality
          (fun g => closure_type_part_stable (closure_at g))
          (fun g => meet_closure_type_ok mh (closure_at g) Hkeeps Hsound))
       (map_meet_ok_for _ _ Pos.eqb _ _ (meet mh) pos_eqb_decides_equality
          (fun w => present_part_stable (value_slot_at w))
          (fun w => meet_type_present_ok mh Hkeeps Hsound (value_slot_at w)))
       (fun tenv a b c Ha Hb Hc =>
          proj2 (closures_entry_fits_maps tenv a b c) (conj Ha (conj Hb Hc)))
       (fun statics valuation a b c =>
          closures_entry_holds_maps v slot functions value_slots statics
            valuation a b c)
       Henv Hf1 Hf2 Hc1 Hc2 Hv1 Hv2 Hmeet).
Qed.

Lemma meet_closures_entry_slots_for :
  for_validity meet_closures_entry_slots_statement.
Proof.
  intros mh menv menv' entry1 entry2 answer r Hmeet Hanswer.
  destruct entry1 as [function_types1 closure_types1 value_slot_types1].
  destruct entry2 as [function_types2 closure_types2 value_slot_types2].
  cbn [meet_closures_entry] in Hmeet.
  rewrite combine_results3_eq in Hmeet. unfold combine_results2 in Hmeet.
  destruct (map_meet Pos.eqb meet_function_type menv function_types1
              function_types2)
    as [ [ r_a | res_a menv1 ] | ] eqn:Hmeet_a; try discriminate Hmeet.
  cbn in Hmeet.
  destruct (meet_product_function_slot_indexed mh menv1 closure_types1
              closure_types2)
    as [ [ r_b | res_b menv2 ] | ] eqn:Hmeet_b; try discriminate Hmeet.
  cbn in Hmeet.
  destruct (meet_product_value_slot_indexed mh menv2 value_slot_types1
              value_slot_types2)
    as [ [ r_c | res_c menv3 ] | ] eqn:Hmeet_c; try discriminate Hmeet.
  cbn in Hmeet. injection Hmeet as <- <-.
  apply (combine_answer (fun a bc => Closures_entry a (fst bc) (snd bc)))
    in Hanswer as (a & bc & -> & Ha & Hbc).
  apply (combine_answer pair) in Hbc as (b & c & -> & Hb & Hc).
  cbn [fst snd]. split; [ | split ].
  - intros g.
    exact (map_meet_keys_for _ _ Pos.eqb meet_function_type menv menv1
             function_types1 function_types2 a res_a g
             pos_eqb_decides_equality Hmeet_a Ha).
  - intros g.
    exact (map_meet_keys_for _ _ Pos.eqb (meet mh) menv1 menv2 closure_types1
             closure_types2 b res_b g pos_eqb_decides_equality Hmeet_b Hb).
  - intros w.
    exact (map_meet_keys_for _ _ Pos.eqb (meet mh) menv2 menv3 value_slot_types1
             value_slot_types2 c res_c w pos_eqb_decides_equality Hmeet_c Hc).
Qed.

End Validity.

Theorem extract_value_answer :
  extract_value_answer_statement.
Proof.
  first
    [ exact (@extract_value_answer_for standard_validity)
    | exact extract_value_answer_for ].
Qed.

Theorem combine_results3_ok :
  combine_results3_ok_statement.
Proof.
  first
    [ exact (@combine_results3_ok_for standard_validity)
    | exact combine_results3_ok_for ].
Qed.

Theorem map_meet_keys :
  map_meet_keys_statement.
Proof.
  first
    [ exact (@map_meet_keys_for standard_validity)
    | exact map_meet_keys_for ].
Qed.

Theorem map_meet_ok :
  map_meet_ok_statement.
Proof.
  first
    [ exact (@map_meet_ok_for standard_validity)
    | exact map_meet_ok_for ].
Qed.

Theorem product_satisfied_fields :
  product_satisfied_fields_statement.
Proof.
  first
    [ exact (@product_satisfied_fields_for standard_validity)
    | exact product_satisfied_fields_for ].
Qed.

Theorem meet_int_indexed_product_ok :
  meet_int_indexed_product_ok_statement.
Proof.
  first
    [ exact (@meet_int_indexed_product_ok_for standard_validity)
    | exact meet_int_indexed_product_ok_for ].
Qed.

Theorem meet_int_indexed_product_length :
  meet_int_indexed_product_length_statement.
Proof.
  first
    [ exact (@meet_int_indexed_product_length_for standard_validity)
    | exact meet_int_indexed_product_length_for ].
Qed.

Theorem meet_function_type_ok :
  meet_function_type_ok_statement.
Proof.
  first
    [ exact (@meet_function_type_ok_for standard_validity)
    | exact meet_function_type_ok_for ].
Qed.

Theorem meet_closures_entry_ok :
  meet_closures_entry_ok_statement.
Proof.
  first
    [ exact (@meet_closures_entry_ok_for standard_validity)
    | exact meet_closures_entry_ok_for ].
Qed.

Theorem meet_closures_entry_slots :
  meet_closures_entry_slots_statement.
Proof.
  first
    [ exact (@meet_closures_entry_slots_for standard_validity)
    | exact meet_closures_entry_slots_for ].
Qed.

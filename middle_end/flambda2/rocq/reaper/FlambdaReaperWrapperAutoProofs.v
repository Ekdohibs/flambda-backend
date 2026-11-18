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


(* The proofs of FlambdaReaperWrapperAutoSpec.v. Nothing here needs
   review. *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaWellFormedSpec FlambdaReaperGraphSpec
  FlambdaReaperTraversal FlambdaReaperSolution FlambdaReaperSolutionProofs
  FlambdaReaperRebuild FlambdaReaperCallingConventions FlambdaReaperUnboxing
  FlambdaReaperLeast FlambdaReaperFlowSpec FlambdaReaperLeastSpec
  FlambdaReaperWrapperSpec FlambdaReaperWrapperShapeSpec
  FlambdaReaperWrapperAutoSpec.
From OxCamlFlambda2 Require FlambdaMapProofs FlambdaReaperFlowProofs
  FlambdaReaperLeastProofs FlambdaReaperWrapperProofs
  FlambdaReaperWrapperShapeProofs FlambdaReaperCallingConventionsProofs.
Import ListNotations.

(* ======================================================================== *)
(* The invariant of the traversal                                           *)
(* ======================================================================== *)

(* The witness of known arity of a code dep is made by [prepare_code],
   and nothing but [create_known_arity_call_witness] puts anything into
   it. The invariant of the accumulator ([kinv]), as [winv] of
   [FlambdaReaperLeastProofs.AutoWidgets], with the witnesses of known
   arity of the code deps in place of the widgets: they are made nodes,
   and the facts that put something into one are those of its code; the
   nodes that facts put something into are made before the counter; the
   nodes of the deps into which the delayed facts put something are not
   such witnesses ([good]); and no fact is of a constructor of the field
   of known arity, which only the delayed facts make. A step adds facts
   that put something only into nodes that are not made, good at the
   start of the step or made by it ([ext]); [prepare_code] is the one
   step that makes a code dep ([tstep_prepare_code]). *)
Module KnownWitnesses.

Import FlambdaReaperLeastProofs.AutoWidgets.

Abbreviation known := (Field_call_witness Known_arity_code_pointer).
Abbreviation env_conts := FlambdaReaperTraversal.env_conts.
Abbreviation kw := cd_known_arity_call_witness.

(* [x] is the witness of known arity of a code dep of [a]. *)
Definition kwit (a : acc) (x : node) : Prop :=
  ∃ id cd, (id, cd) ∈ acc_code_deps a ∧ x = kw cd.

(* What [create_known_arity_call_witness] puts into the witness of known
   arity of the code dep [cd] of [id]. *)
Definition known_fact (id : code_id) (cd : code_dep) (fct : fact) : Prop :=
  match fct with
  | Fact_constructor _ (Field_return_of_call (Normal i)) r =>
    nth_error (cd_return cd) i = Some r
  | Fact_constructor _ (Field_return_of_call Exn) _ => True
  | Fact_constructor _ Field_code_id_of_call_witness r => r = Node_code_id id
  | Fact_parameter _ _ p => p <> kw cd
  | _ => False
  end.

(* [x] is made before [a] and is not a witness of known arity of it. *)
Definition good (a : acc) (x : node) : Prop :=
  made_lt x (acc_next a) ∧ ~ kwit a x.

(* [x] is not made, or good in [a0], or made from [a0] to [a]. *)
Definition ok_node (a0 a : acc) (x : node) : Prop :=
  not_made x ∨ good a0 x
  ∨ ∃ k, x = Node_made k ∧ (acc_next a0 <= k)%positive
         ∧ (k < acc_next a)%positive.

Record kinv (a : acc) : Prop := {
  ki_into : ∀ fct x,
    fct ∈ acc_facts a -> x ∈ into_nodes fct -> made_lt x (acc_next a);
  ki_made : ∀ id cd,
    (id, cd) ∈ acc_code_deps a ->
    ∃ k, kw cd = Node_made k ∧ (k < acc_next a)%positive;
  ki_known : ∀ id cd fct,
    (id, cd) ∈ acc_code_deps a -> fct ∈ acc_facts a ->
    kw cd ∈ into_nodes fct -> known_fact id cd fct;
  ki_constructor : ∀ b from, Fact_constructor b known from ∉ acc_facts a;
  ki_code_deps : ∀ id cd x,
    (id, cd) ∈ acc_code_deps a -> x ∈ code_dep_nodes cd -> good a x;
  ki_apply_deps : ∀ d,
    d ∈ acc_apply_deps a -> good a (ad_apply_call_witness d);
  ki_closure_deps : ∀ d,
    d ∈ acc_closure_deps a -> good a (cld_let_bound_name_of_the_closure d)
}.

Definition env_good (env : traverse_env) (a : acc) : Prop :=
  ∀ k params x, (k, params) ∈ env_conts env -> x ∈ params -> good a x.

(* The witnesses of known arity of [a'] are those of [a] and nodes made
   from [a]. *)
Definition grows (a a' : acc) : Prop :=
  (acc_next a <= acc_next a')%positive
  ∧ ∀ w, kwit a' w ->
      kwit a w ∨ ∃ k, w = Node_made k ∧ (acc_next a <= k)%positive.

(* From [a] to [a'], facts are added that put something into nodes that
   are not made, good in [a0] or made from [a0], and no constructor of
   the field of known arity; and deps of such nodes, but no code dep. *)
Record ext (a0 a a' : acc) : Prop := {
  ext_next : (acc_next a <= acc_next a')%positive;
  ext_facts : ∃ L, acc_facts a' = L ++ acc_facts a
    ∧ (∀ fct x, fct ∈ L -> x ∈ into_nodes fct -> ok_node a0 a' x)
    ∧ (∀ b from, Fact_constructor b known from ∉ L);
  ext_code_deps : acc_code_deps a' = acc_code_deps a;
  ext_apply_deps : ∀ d, d ∈ acc_apply_deps a' ->
    d ∈ acc_apply_deps a ∨ ok_node a0 a' (ad_apply_call_witness d);
  ext_closure_deps : ∀ d, d ∈ acc_closure_deps a' ->
    d ∈ acc_closure_deps a
    ∨ ok_node a0 a' (cld_let_bound_name_of_the_closure d)
}.

(* ------------------------------------------------------------------------ *)
(* Basic lemmas                                                             *)
(* ------------------------------------------------------------------------ *)

Lemma ok_node_mono (a0 a a' : acc) (x : node) :
  (acc_next a <= acc_next a')%positive -> ok_node a0 a x -> ok_node a0 a' x.
Proof.
  intros Hle [H | [H | (k & -> & H1 & H2)]]; [ left | right; left | ];
    auto.
  right; right. exists k. split; [ reflexivity | lia ].
Qed.

Lemma ok_simple (a0 a : acc) (s : simple) : ok_node a0 a (node_of_simple s).
Proof. left. apply not_made_simple. Qed.

Lemma ok_not_made (a0 a : acc) (x : node) : not_made x -> ok_node a0 a x.
Proof. intros H. left. exact H. Qed.

Lemma ok_good (a0 a : acc) (x : node) : good a0 x -> ok_node a0 a x.
Proof. intros H. right; left. exact H. Qed.

Lemma ok_made (a0 a : acc) (k : positive) :
  (acc_next a0 <= k)%positive -> (k < acc_next a)%positive ->
  ok_node a0 a (Node_made k).
Proof. intros H1 H2. right; right. exists k. auto. Qed.

Lemma ext_trans (a0 a a1 a2 : acc) :
  ext a0 a a1 -> ext a0 a1 a2 -> ext a0 a a2.
Proof.
  intros [Hn1 (L1 & Hf1 & Hi1 & Hw1) Hc1 Hap1 Hcl1]
    [Hn2 (L2 & Hf2 & Hi2 & Hw2) Hc2 Hap2 Hcl2].
  constructor; [ lia | | congruence | | ].
  - exists (L2 ++ L1). split; [ rewrite Hf2, Hf1, app_assoc; reflexivity | ].
    split.
    + intros fct x Hin Hx. apply in_app_or in Hin as [Hin | Hin];
        [ exact (Hi2 _ _ Hin Hx) | ].
      exact (ok_node_mono _ _ _ _ Hn2 (Hi1 _ _ Hin Hx)).
    + intros w n Hin. apply in_app_or in Hin as [Hin | Hin];
        [ exact (Hw2 _ _ Hin) | exact (Hw1 _ _ Hin) ].
  - intros d Hin. destruct (Hap2 _ Hin) as [Hin' | H]; [ | right; auto ].
    destruct (Hap1 _ Hin') as [H' | H']; [ left; auto | right ].
    exact (ok_node_mono _ _ _ _ Hn2 H').
  - intros d Hin. destruct (Hcl2 _ Hin) as [Hin' | H]; [ | right; auto ].
    destruct (Hcl1 _ Hin') as [H' | H']; [ left; auto | right ].
    exact (ok_node_mono _ _ _ _ Hn2 H').
Qed.

Lemma ext_same (a0 a a' : acc) :
  (acc_next a <= acc_next a')%positive ->
  acc_facts a' = acc_facts a -> acc_code_deps a' = acc_code_deps a ->
  acc_apply_deps a' = acc_apply_deps a ->
  acc_closure_deps a' = acc_closure_deps a -> ext a0 a a'.
Proof.
  intros Hn Hf Hc Hap Hcl. constructor; auto.
  - exists []. rewrite Hf. split; [ reflexivity | split ];
      [ intros fct x [] | intros w n [] ].
  - intros d Hin. left. rewrite <- Hap. exact Hin.
  - intros d Hin. left. rewrite <- Hcl. exact Hin.
Qed.

Lemma ext_refl (a0 a : acc) : ext a0 a a.
Proof. apply ext_same; (reflexivity || lia). Qed.

Lemma ext_add_facts (a0 a : acc) (L : list fact) :
  (∀ fct x, fct ∈ L -> x ∈ into_nodes fct -> ok_node a0 a x) ->
  (∀ b from, Fact_constructor b known from ∉ L) ->
  ext a0 a (add_facts L a).
Proof.
  intros Hi Hw. constructor; cbn [acc_next acc_facts acc_code_deps
    acc_apply_deps acc_closure_deps add_facts]; [ lia | | | | ].
  - exists L. split; [ reflexivity | split; assumption ].
  - reflexivity.
  - intros d Hin. left. exact Hin.
  - intros d Hin. left. exact Hin.
Qed.

Lemma ext_add_fact (a0 a : acc) (fct : fact) :
  (∀ x, x ∈ into_nodes fct -> ok_node a0 a x) ->
  (∀ b from, fct <> Fact_constructor b known from) ->
  ext a0 a (add_fact fct a).
Proof.
  intros Hi Hw. apply ext_add_facts.
  - intros fct' x [<- | []] Hx. exact (Hi x Hx).
  - intros w n [H | []]. exact (Hw w n H).
Qed.

Lemma ext_fresh (a0 a : acc) : ext a0 a (snd (fresh a)).
Proof. apply ext_same; cbn; (reflexivity || lia). Qed.

Lemma ok_fresh (a0 a : acc) :
  (acc_next a0 <= acc_next a)%positive ->
  ok_node a0 (snd (fresh a)) (fst (fresh a)).
Proof. intros H. apply ok_made; cbn; lia. Qed.

Lemma ext_note_simple (a0 : acc) (s : simple) (a : acc) :
  ext a0 a (note_simple s a).
Proof. destruct s; apply ext_same; cbn; (reflexivity || lia). Qed.

Lemma ext_note_simples (a0 : acc) (l : list simple) (a : acc) :
  ext a0 a (note_simples l a).
Proof.
  destruct (note_simples_fields l a) as (H1 & H2 & H3 & H4 & H5).
  apply ext_same; auto. rewrite H1. lia.
Qed.

Lemma ext_push_apply_dep (a0 : acc) (d : apply_dep) (a : acc) :
  ok_node a0 a (ad_apply_call_witness d) -> ext a0 a (push_apply_dep d a).
Proof.
  intros H. constructor; cbn; [ lia | | reflexivity | | ].
  - exists []. split; [ reflexivity | split ];
      [ intros fct x [] | intros w n [] ].
  - intros d' [<- | Hin]; [ right; exact H | left; exact Hin ].
  - intros d' Hin. left. exact Hin.
Qed.

Lemma ext_push_closure_dep (a0 : acc) (d : closure_dep) (a : acc) :
  ok_node a0 a (cld_let_bound_name_of_the_closure d) ->
  ext a0 a (push_closure_dep d a).
Proof.
  intros H. constructor; cbn; [ lia | | reflexivity | | ].
  - exists []. split; [ reflexivity | split ];
      [ intros fct x [] | intros w n [] ].
  - intros d' Hin. left. exact Hin.
  - intros d' [<- | Hin]; [ right; exact H | left; exact Hin ].
Qed.

Lemma ext_fold {A : Type} (a0 : acc) (f : acc -> A -> acc) (l : list A)
    (a : acc) :
  (∀ y b, y ∈ l -> (acc_next a <= acc_next b)%positive -> ext a0 b (f b y)) ->
  ext a0 a (List.fold_left f l a).
Proof.
  revert a. induction l as [ | y l IH]; intros a H; cbn [List.fold_left].
  - apply ext_refl.
  - pose proof (H y a (or_introl eq_refl) (Pos.le_refl _)) as H1.
    apply (ext_trans _ _ _ _ H1). apply IH.
    intros y' b Hy' Hb. apply H; [ right; exact Hy' | ].
    pose proof (ext_next _ _ _ H1). lia.
Qed.

Lemma kwit_made (a : acc) (x : node) :
  kinv a -> kwit a x -> ∃ k, x = Node_made k ∧ (k < acc_next a)%positive.
Proof. intros Hinv (id & cd & Hin & ->). exact (ki_made a Hinv _ _ Hin). Qed.

Lemma ok_close (a0 a' : acc) (x : node) :
  kinv a0 -> ext a0 a0 a' -> ok_node a0 a' x -> good a' x.
Proof.
  intros Hinv [Hn _ Hc _ _].
  assert (Hk : ∀ w, kwit a' w -> kwit a0 w).
  { intros w (id & cd & Hin & ->). rewrite Hc in Hin.
    exists id, cd. split; [ exact Hin | reflexivity ]. }
  intros [Hx | [[Hlt Hnw] | (k & -> & H1 & H2)]].
  - split; [ intros k -> ; destruct (Hx k eq_refl) | ].
    intros Hwx. destruct (kwit_made a0 x Hinv (Hk x Hwx)) as (k & -> & _).
    exact (Hx k eq_refl).
  - split; [ intros k Hk'; specialize (Hlt k Hk'); lia | ].
    intros Hwx. exact (Hnw (Hk x Hwx)).
  - split; [ intros k' Hk'; injection Hk' as <-; exact H2 | ].
    intros Hwx. destruct (kwit_made a0 _ Hinv (Hk _ Hwx)) as (k' & Hk' & Hlt).
    injection Hk' as <-. lia.
Qed.

Lemma ext_close (a0 a' : acc) :
  kinv a0 -> ext a0 a0 a' -> kinv a' ∧ grows a0 a'.
Proof.
  intros Hinv He. pose proof (fun x => ok_close a0 a' x Hinv He) as Hgood.
  destruct He as [Hn (L & Hf & Hi & Hw) Hc Hap Hcl].
  assert (Hk : ∀ w, kwit a' w -> kwit a0 w).
  { intros w (id & cd & Hin & ->). rewrite Hc in Hin.
    exists id, cd. split; [ exact Hin | reflexivity ]. }
  assert (Hgood0 : ∀ x, good a0 x -> good a' x).
  { intros x Hx. apply Hgood. right; left. exact Hx. }
  split; [ constructor | split; [ exact Hn | ] ].
  - intros fct x Hin Hx. rewrite Hf in Hin.
    apply in_app_or in Hin as [Hin | Hin].
    + exact (proj1 (Hgood x (Hi _ _ Hin Hx))).
    + intros k Hk'. specialize (ki_into a0 Hinv _ _ Hin Hx k Hk'). lia.
  - intros id cd Hin. rewrite Hc in Hin.
    destruct (ki_made a0 Hinv _ _ Hin) as (k & Hk' & Hlt).
    exists k. split; [ exact Hk' | lia ].
  - intros id cd fct Hin Hfct Hx. rewrite Hf in Hfct.
    apply in_app_or in Hfct as [Hfct | Hfct].
    + exfalso. destruct (Hgood _ (Hi _ _ Hfct Hx)) as [_ Hnk].
      apply Hnk. exists id, cd. split; [ exact Hin | reflexivity ].
    + rewrite Hc in Hin. exact (ki_known a0 Hinv _ _ _ Hin Hfct Hx).
  - intros b from Hin. rewrite Hf in Hin.
    apply in_app_or in Hin as [Hin | Hin];
      [ exact (Hw _ _ Hin) | exact (ki_constructor a0 Hinv _ _ Hin) ].
  - intros id cd x Hin Hx. rewrite Hc in Hin.
    exact (Hgood0 _ (ki_code_deps a0 Hinv _ _ _ Hin Hx)).
  - intros d Hin. destruct (Hap _ Hin) as [Hin' | H].
    + exact (Hgood0 _ (ki_apply_deps a0 Hinv _ Hin')).
    + exact (Hgood _ H).
  - intros d Hin. destruct (Hcl _ Hin) as [Hin' | H].
    + exact (Hgood0 _ (ki_closure_deps a0 Hinv _ Hin')).
    + exact (Hgood _ H).
  - intros w Hwx. left. exact (Hk w Hwx).
Qed.

Lemma grows_refl (a : acc) : grows a a.
Proof. split; [ lia | intros w H; left; exact H ]. Qed.

Lemma grows_trans (a a1 a2 : acc) : grows a a1 -> grows a1 a2 -> grows a a2.
Proof.
  intros [Hn1 Hw1] [Hn2 Hw2]. split; [ lia | ].
  intros w Hw. destruct (Hw2 w Hw) as [H | (k & -> & Hk)].
  - destruct (Hw1 _ H) as [H' | (k & -> & Hk)]; [ left; exact H' | ].
    right. exists k. split; [ reflexivity | lia ].
  - right. exists k. split; [ reflexivity | lia ].
Qed.

Lemma good_grows (a a' : acc) (x : node) :
  grows a a' -> good a x -> good a' x.
Proof.
  intros [Hn Hw] [Hlt Hnw]. split.
  - intros k Hk. specialize (Hlt k Hk). lia.
  - intros Hwx. destruct (Hw x Hwx) as [H | (k & -> & Hk)];
      [ exact (Hnw H) | ].
    specialize (Hlt k eq_refl). lia.
Qed.

Lemma env_good_grows (env : traverse_env) (a a' : acc) :
  grows a a' -> env_good env a -> env_good env a'.
Proof.
  intros Hg H k params x Hin Hx. exact (good_grows _ _ _ Hg (H _ _ _ Hin Hx)).
Qed.

(* A step of the traversal, from an accumulator of the invariant whose
   continuations in scope are good. *)
Definition tstep (env : traverse_env) (a a' : acc) : Prop :=
  kinv a -> env_good env a -> kinv a' ∧ grows a a'.

Lemma tstep_refl (env : traverse_env) (a : acc) : tstep env a a.
Proof. intros H _. split; [ exact H | apply grows_refl ]. Qed.

Lemma tstep_trans (env : traverse_env) (a a1 a2 : acc) :
  tstep env a a1 -> tstep env a1 a2 -> tstep env a a2.
Proof.
  intros H1 H2 Hinv Henv. destruct (H1 Hinv Henv) as [Hinv1 Hg1].
  destruct (H2 Hinv1 (env_good_grows _ _ _ Hg1 Henv)) as [Hinv2 Hg2].
  split; [ exact Hinv2 | exact (grows_trans _ _ _ Hg1 Hg2) ].
Qed.

Lemma tstep_ext (env : traverse_env) (a a' : acc) :
  (kinv a -> env_good env a -> ext a a a') -> tstep env a a'.
Proof. intros H Hinv Henv. exact (ext_close a a' Hinv (H Hinv Henv)). Qed.

Lemma tstep_fold {A : Type} (env : traverse_env) (f : acc -> A -> acc)
    (l : list A) (a : acc) :
  (∀ y b, y ∈ l -> tstep env b (f b y)) -> tstep env a (List.fold_left f l a).
Proof.
  revert a. induction l as [ | y l IH]; intros a H; cbn [fold_left].
  - apply tstep_refl.
  - apply (tstep_trans _ _ (f a y)); [ apply H; left; reflexivity | ].
    apply IH. intros y' b Hy'. apply H. right. exact Hy'.
Qed.

(* ------------------------------------------------------------------------ *)
(* The steps of the traversal                                               *)
(* ------------------------------------------------------------------------ *)

Ltac into_split H :=
  repeat match type of H with
  | In _ (_ ++ _) => apply in_app_or in H; destruct H as [H | H]
  | In _ (map _ _) =>
    let y := fresh "y" in let Hy := fresh "Hy" in
    apply in_map_iff in H; destruct H as [y [H Hy]]
  | In _ (flat_map _ _) =>
    let y := fresh "y" in let Hy := fresh "Hy" in
    apply in_flat_map in H; destruct H as [y [Hy H]]
  | In _ (_ :: _) => destruct H as [H | H]
  | In _ [] => destruct H
  | In _ (match ?t with _ => _ end) => destruct t
  end.

Ltac into_close H Hx :=
  repeat match goal with y : prod _ _ |- _ => destruct y end;
  subst; cbn [into_nodes] in Hx;
  repeat match type of Hx with
  | In _ (_ :: _) => destruct Hx as [Hx | Hx]
  | In _ [] => destruct Hx
  end;
  subst.

Ltac no_known :=
  let w := fresh "w" in let n := fresh "n" in let H := fresh "H" in
  intros w n H; into_split H;
  repeat match goal with y : prod _ _ |- _ => destruct y end;
  try discriminate.

Ltac ok_tac :=
  first
    [ assumption
    | apply ok_simple
    | apply ok_not_made; intros ?k; discriminate
    | apply ok_made; cbn [acc_next]; lia
    | eapply ok_node_mono; [ | eassumption ]; cbn [acc_next]; lia ].

Lemma ext_cond_any_usage (a0 : acc) (env : traverse_env) (s : simple)
    (a : acc) :
  ext a0 a (cond_any_usage env s a).
Proof.
  unfold cond_any_usage.
  apply (ext_trans _ _ (note_simple s a)); [ apply ext_note_simple | ].
  destruct (env_current_code_id env) as [id | ]; apply ext_add_fact;
    try (intros ? ? ?; discriminate).
  - intros x [<- | []]. ok_tac.
  - intros x [].
Qed.

Lemma ext_cond_alias (a0 : acc) (env : traverse_env) (from to_ : node)
    (a : acc) :
  ok_node a0 a to_ -> ext a0 a (cond_alias env from to_ a).
Proof.
  intros H. unfold cond_alias.
  destruct (env_current_code_id env) as [id | ]; apply ext_add_fact;
    try (intros ? ? ?; discriminate); intros x [<- | []]; exact H.
Qed.


Lemma ext_fresh_list (a0 : acc) (n : nat) (a : acc) :
  (acc_next a0 <= acc_next a)%positive ->
  ext a0 a (snd (fresh_list n a))
  ∧ ∀ x, x ∈ fst (fresh_list n a) -> ok_node a0 (snd (fresh_list n a)) x.
Proof.
  intros Hle. destruct (fresh_list_spec n a) as (H1 & H2 & H3 & H4 & H5 & H6).
  split; [ apply ext_same; auto | ].
  intros x Hx. destruct (H6 x Hx) as (m & -> & Hm1 & Hm2).
  apply ok_made; lia.
Qed.


Lemma ext_find_cont (a0 : acc) (env : traverse_env) (k : continuation)
    (count : nat) (a : acc) :
  env_good env a0 -> (acc_next a0 <= acc_next a)%positive ->
  ext a0 a (snd (find_cont env k count a))
  ∧ ∀ x, x ∈ fst (find_cont env k count a) ->
      ok_node a0 (snd (find_cont env k count a)) x.
Proof.
  intros Henv Hle. unfold find_cont.
  destruct (lookup k (env_conts env)) as [params | ] eqn:E.
  - cbn [fst snd]. split; [ apply ext_refl | ].
    intros x Hx. apply ok_good.
    exact (Henv k params x (lookup_in_conts _ _ _ E) Hx).
  - destruct (ext_fresh_list a0 count a Hle) as [H1 H2].
    destruct (fresh_list count a) as [params b] eqn:Eb.
    cbn [fst snd] in *. split.
    + apply (ext_trans _ _ _ _ H1). apply ext_add_facts.
      * intros fct x Hin Hx. into_split Hin. into_close Hin Hx.
      * no_known.
    + intros x Hx. exact (H2 x Hx).
Qed.

Lemma ext_block_deps (a0 : acc) (base : node) (fields : list simple)
    (a : acc) :
  ok_node a0 a base -> ext a0 a (block_deps base fields a).
Proof.
  intros H. unfold block_deps.
  destruct (note_simples_fields fields a) as (H1 & _).
  apply (ext_trans _ _ (note_simples fields a));
    [ apply ext_note_simples | ].
  apply ext_add_facts.
  - intros fct x Hin Hx. into_split Hin; into_close Hin Hx;
      eapply ok_node_mono; try eassumption; rewrite H1; lia.
  - no_known.
Qed.

Lemma ext_traverse_prim (a0 : acc) (env : traverse_env) (x : variable)
    (p : prim) (args : list simple) (a : acc) :
  ext a0 a (traverse_prim env x p args a).
Proof.
  unfold traverse_prim.
  assert (Hacc : ∀ f s b,
             ext a0 b (add_fact (Fact_accessor (Node_variable x) f
                                   (node_of_simple s)) (note_simple s b))).
  { intros f s b. apply (ext_trans _ _ (note_simple s b));
      [ apply ext_note_simple | ].
    apply ext_add_fact.
    - intros y [<- | []]. ok_tac.
    - intros ? ? ?; discriminate. }
  assert (Hdefault :
    ext a0 a
      (add_facts
         (Fact_any_source (Node_variable x)
          :: map (fun from => Fact_use (Node_variable x) from)
               (free_names_of_simples args))
         (if prim_has_arbitrary_effects p
          then cond_any_usage env (Var x) a else a))).
  { apply (ext_trans _ _ (if prim_has_arbitrary_effects p
                          then cond_any_usage env (Var x) a else a)).
    - destruct (prim_has_arbitrary_effects p);
        [ apply ext_cond_any_usage | apply ext_refl ].
    - apply ext_add_facts.
      + intros fct y Hin Hy. into_split Hin; into_close Hin Hy; ok_tac.
      + no_known. }
  destruct p as [ u | b | v | h | r ];
    [ destruct u | destruct b | destruct v | destruct h | destruct r ];
    repeat match goal with
    | |- ext _ _ (match ?t with _ => _ end) => destruct t
    end;
    try exact Hdefault;
    try (apply ext_block_deps; ok_tac);
    try apply Hacc;
    try apply ext_refl.
  - match goal with
    | |- ext _ _ (add_fact _ ?b) => apply (ext_trans _ _ b)
    end; [ apply Hacc | ].
    apply ext_add_fact; [ intros y [<- | []]; ok_tac | ].
    intros ? ? ?; discriminate.
  - apply (ext_trans _ _ (note_simple s a)); [ apply ext_note_simple | ].
    apply ext_add_fact; [ intros y [<- | []]; ok_tac | ].
    intros ? ? ?; discriminate.
Qed.

Lemma ext_record_set (a0 : acc) (env : traverse_env)
    (closures : list (node * option code_id))
    (function_slots : list (function_slot * node))
    (value_slots : list (value_slot * simple)) (a : acc) :
  (∀ base code, (base, code) ∈ closures -> ok_node a0 a base) ->
  ext a0 a (record_set_of_closures_deps env closures function_slots
              value_slots a).
Proof.
  intros H. unfold record_set_of_closures_deps. apply ext_fold.
  intros [base code] b Hin Hle. cbv beta iota.
  assert (Hb : ok_node a0 b base)
    by exact (ok_node_mono _ _ _ _ Hle (H _ _ Hin)).
  set (b1 := match code with
             | Some id => push_closure_dep _ b
             | None => b
             end).
  assert (H1 : ext a0 b b1).
  { subst b1. destruct code; [ apply ext_push_closure_dep; exact Hb
                             | apply ext_refl ]. }
  apply (ext_trans _ _ _ _ H1).
  match goal with
  | |- ext _ _ (add_facts _ (note_simples ?l _)) =>
    destruct (note_simples_fields l b1) as (Hn & _);
    apply (ext_trans _ _ (note_simples l b1)); [ apply ext_note_simples | ]
  end.
  apply ext_add_facts.
  - intros fct x Hfct Hx. into_split Hfct; into_close Hfct Hx;
      apply (ok_node_mono _ b); [ | exact Hb | | exact Hb ];
      rewrite Hn; exact (ext_next _ _ _ H1).
  - no_known.
Qed.

Lemma ext_closures_of_set (a0 : acc) (x : variable) (slot : function_slot)
    (others : list (function_slot * variable))
    (decls : list (function_slot * code_id)) (a : acc) :
  (acc_next a0 <= acc_next a)%positive ->
  ext a0 a (snd (closures_of_set x slot others decls a))
  ∧ ∀ f name id, (f, name, id) ∈ fst (closures_of_set x slot others decls a) ->
      ok_node a0 (snd (closures_of_set x slot others decls a)) name.
Proof.
  revert a. induction decls as [ | [f id] decls IH]; intros a Hle;
    cbn [closures_of_set].
  - cbn [fst snd]. split; [ apply ext_refl | intros f name id [] ].
  - match goal with
    | |- context [let '(_, _) := ?t in _] =>
      assert (Ht : ext a0 a (snd t) ∧ ok_node a0 (snd t) (fst t));
      [ | destruct t as [name b] eqn:E ]
    end.
    { destruct (Pos.eqb f slot);
        [ cbn; split; [ apply ext_refl | ok_tac ] | ].
      destruct (lookup f others);
        [ cbn; split; [ apply ext_refl | ok_tac ] | ].
      split; [ apply ext_fresh | apply ok_fresh; exact Hle ]. }
    cbn [fst snd] in Ht. destruct Ht as [H1 Hname].
    destruct (IH b) as [H2 H3]; [ pose proof (ext_next _ _ _ H1); lia | ].
    destruct (closures_of_set x slot others decls b) as [rest c] eqn:E2.
    cbn [fst snd] in *. split; [ exact (ext_trans _ _ _ _ H1 H2) | ].
    intros f' name' id' [Heq | Hin].
    + injection Heq as <- <- <-.
      exact (ok_node_mono _ _ _ _ (ext_next _ _ _ H2) Hname).
    + exact (H3 _ _ _ Hin).
Qed.

Lemma ext_traverse_closure_of_set (a0 : acc) (env : traverse_env)
    (x : variable) (slot : function_slot)
    (function_decls : list (function_slot * code_id))
    (value_slots : list (value_slot * simple)) (a : acc) :
  (acc_next a0 <= acc_next a)%positive ->
  ext a0 a (traverse_closure_of_set env x slot function_decls value_slots a).
Proof.
  intros Hle. unfold traverse_closure_of_set. cbv zeta.
  destruct (among slot _); [ | apply ext_refl ].
  match goal with
  | |- context [closures_of_set ?x ?s ?o ?d a] =>
    destruct (ext_closures_of_set a0 x s o d a Hle) as [H1 H2];
    destruct (closures_of_set x s o d a) as [closures b] eqn:E
  end.
  cbn [fst snd] in *. apply (ext_trans _ _ _ _ H1).
  apply ext_record_set. intros base code Hin.
  apply in_map_iff in Hin. destruct Hin as [[[f name] id] [Heq Hin]].
  injection Heq as <- _. exact (H2 _ _ _ Hin).
Qed.

Lemma ext_traverse_named (a0 : acc) (env : traverse_env) (x : variable)
    (defining_expr : named) (a : acc) :
  (acc_next a0 <= acc_next a)%positive ->
  ext a0 a (traverse_named env x defining_expr a).
Proof.
  intros Hle. unfold traverse_named.
  destruct defining_expr as [ s | p args | id vs | slot decls vs | w c
                            | m w c | mf mt c ].
  - apply (ext_trans _ _ (note_simple s a)); [ apply ext_note_simple | ].
    apply ext_add_fact; [ intros y [<- | []]; ok_tac | ].
    intros ? ? ?; discriminate.
  - apply ext_traverse_prim.
  - apply ext_record_set. intros base code [Heq | []].
    injection Heq as <- _. ok_tac.
  - apply ext_traverse_closure_of_set. exact Hle.
  - apply (ext_trans _ _ (note_simple c a)); [ apply ext_note_simple | ].
    apply ext_add_fact; [ intros y [<- | []]; ok_tac | ].
    intros ? ? ?; discriminate.
  - apply (ext_trans _ _ (note_simple c a)); [ apply ext_note_simple | ].
    apply ext_add_fact; [ intros y [<- | []]; ok_tac | ].
    intros ? ? ?; discriminate.
  - apply (ext_trans _ _ (note_simple c a)); [ apply ext_note_simple | ].
    apply ext_add_fact; [ intros y [<- | []]; ok_tac | ].
    intros ? ? ?; discriminate.
Qed.

Lemma ext_traverse_static_binding (a0 : acc) (env : traverse_env)
    (binding : static_binding) (a : acc) :
  ext a0 a (traverse_static_binding env binding a).
Proof.
  unfold traverse_static_binding.
  destruct binding as [ | | s const]; [ apply ext_refl | apply ext_refl | ].
  destruct const as [ id vs | slot syms decls vs | tag fields
                    | contents | str | [ | f fields] ].
  - apply ext_record_set. intros base code [Heq | []].
    injection Heq as <- _. ok_tac.
  - apply ext_record_set. intros base code [Heq | []].
    injection Heq as <- _. ok_tac.
  - apply ext_block_deps. ok_tac.
  - unfold boxed_number_facts. apply ext_add_facts.
    + intros fct y Hin Hy. repeat match goal with
      | H : context [match ?t with _ => _ end] |- _ => destruct t
      end; into_split Hin; into_close Hin Hy; ok_tac.
    + intros w n H. repeat match goal with
      | H : context [match ?t with _ => _ end] |- _ => destruct t
      end; into_split H; discriminate.
  - apply ext_add_fact; [ intros y [<- | []]; ok_tac
                        | intros ? ? ?; discriminate ].
  - apply ext_add_fact; [ intros y [<- | []]; ok_tac
                        | intros ? ? ?; discriminate ].
  - apply ext_block_deps. ok_tac.
Qed.

Lemma fresh_spec (a0 c : acc) (x : node) (b : acc) :
  fresh c = (x, b) -> (acc_next a0 <= acc_next c)%positive ->
  x = Node_made (acc_next c) ∧ acc_next b = Pos.succ (acc_next c)
  ∧ ext a0 c b ∧ ok_node a0 b x.
Proof.
  intros E Hle. pose proof (ext_fresh a0 c) as H1.
  pose proof (ok_fresh a0 c Hle) as H2. rewrite E in H1, H2.
  unfold fresh in E. injection E as <- <-. cbn in *. auto.
Qed.


Ltac nexts :=
  repeat match goal with
  | H : ext _ ?b ?c |- _ =>
    lazymatch goal with
    | _ : (acc_next b <= acc_next c)%positive |- _ => fail
    | _ => pose proof (ext_next _ _ _ H)
    end
  end.

Ltac mono :=
  nexts; eapply ok_node_mono; [ | eassumption ];
  cbn [acc_next add_fact add_facts push_apply_dep push_closure_dep
       add_code_dep] in *; lia.

Ltac fresh_step a0 c x b E Hx :=
  let Hn := fresh "Hn" in let He := fresh "He" in let Ho := fresh "Ho" in
  destruct (fresh c) as [x b] eqn:E;
  destruct (fresh_spec a0 c x b E) as (Hx & Hn & He & Ho);
  [ nexts; lia | ].

Lemma ext_create_known (a0 : acc) (id : code_id) (params returns : list node)
    (exn : node) (a : acc) :
  (acc_next a0 <= acc_next a)%positive ->
  (∀ x, x ∈ params -> ok_node a0 a x) ->
  ext a0 a (snd (create_known_arity_call_witness id params returns exn a))
  ∧ ok_node a0 (snd (create_known_arity_call_witness id params returns exn a))
      (fst (create_known_arity_call_witness id params returns exn a)).
Proof.
  intros Hle Hp. unfold create_known_arity_call_witness.
  fresh_step a0 a w b E Hw. cbv zeta. cbn [fst snd].
  split; [ | exact Ho ].
  apply (ext_trans _ _ _ _ He). apply ext_add_facts; [ | no_known ].
  intros fct x Hin Hx. into_split Hin; into_close Hin Hx;
    try exact Ho.
  eapply ok_node_mono; [ | apply Hp; eapply indexed_in; eassumption ].
  rewrite Hn. lia.
Qed.

Lemma ext_create_tupled (a0 : acc) (id : code_id) (params returns : list node)
    (exn : node) (a : acc) :
  (acc_next a0 <= acc_next a)%positive ->
  (∀ x, x ∈ params -> ok_node a0 a x) ->
  ext a0 a
    (snd (create_unknown_arity_tupled_call_witnesses id params returns exn a))
  ∧ ∀ x,
      x ∈ fst (create_unknown_arity_tupled_call_witnesses id params returns
                 exn a) ->
      ok_node a0
        (snd (create_unknown_arity_tupled_call_witnesses id params returns
                exn a)) x.
Proof.
  intros Hle Hp. unfold create_unknown_arity_tupled_call_witnesses.
  fresh_step a0 a w b E Hw.
  pose proof (ext_next _ _ _ He) as Hb.
  fresh_step a0 b u c E2 Hu.
  pose proof (ext_next _ _ _ He0) as Hc.
  fresh_step a0 c k d E3 Hk.
  cbv zeta. cbn [fst snd].
  assert (Hpd : ∀ x, x ∈ params -> ok_node a0 d x).
  { intros x Hx. eapply ok_node_mono; [ | exact (Hp x Hx) ]. lia. }
  assert (Hwd : ok_node a0 d w) by (eapply ok_node_mono; [ | exact Ho ]; lia).
  assert (Hud : ok_node a0 d u)
    by (eapply ok_node_mono; [ | exact Ho0 ]; lia).
  split.
  - apply (ext_trans _ _ _ _ He). apply (ext_trans _ _ _ _ He0).
    apply (ext_trans _ _ _ _ He1). apply ext_add_facts; [ | no_known ].
    intros fct x Hin Hx. into_split Hin; into_close Hin Hx;
      try ok_tac.
    apply Hpd. eapply indexed_in. eassumption.
  - intros x [<- | []]. cbn [acc_next add_facts].
    exact Hwd.
Qed.

Lemma ext_witness_deps (a0 : acc) (id : code_id) (returns : list node)
    (exn : node) (pw : list (list node * node)) (a : acc) :
  (acc_next a0 <= acc_next a)%positive ->
  (∀ first w, (first, w) ∈ pw ->
     ok_node a0 a w ∧ ∀ x, x ∈ first -> ok_node a0 a x) ->
  ext a0 a (unknown_arity_witness_deps id returns exn pw a).
Proof.
  revert a. induction pw as [ | [first w] rest IH]; intros a Hle Hpw;
    cbn [unknown_arity_witness_deps]; [ apply ext_refl | ].
  destruct (Hpw first w (or_introl eq_refl)) as [Hw Hfirst].
  set (b := add_facts _ a).
  assert (Hbn : acc_next b = acc_next a) by reflexivity.
  assert (H1 : ext a0 a b).
  { apply ext_add_facts; [ | no_known ].
    intros fct x Hin Hx. into_split Hin; into_close Hin Hx; try exact Hw.
    apply Hfirst. eapply indexed_in. eassumption. }
  apply (ext_trans _ _ _ _ H1).
  assert (Hrest : ∀ first' w', (first', w') ∈ rest ->
            ok_node a0 b w' ∧ ∀ x, x ∈ first' -> ok_node a0 b x).
  { intros first' w' Hin. destruct (Hpw _ _ (or_intror Hin)) as [H2 H3].
    split; [ exact H2 | exact H3 ]. }
  destruct rest as [ | [g nw] rest'].
  - apply ext_add_facts; [ | no_known ].
    intros fct x Hin Hx. into_split Hin; into_close Hin Hx; exact Hw.
  - fresh_step a0 b v c E Hv.
    cbv zeta.
    destruct (Hrest g nw (or_introl eq_refl)) as [Hnw _].
    assert (Hcv : ok_node a0 c v) by exact Ho.
    set (d := add_facts _ c).
    assert (Hdn : acc_next d = acc_next c) by reflexivity.
    assert (H2 : ext a0 c d).
    { apply ext_add_facts; [ | no_known ].
      intros fct x Hin Hx. into_split Hin; into_close Hin Hx; try exact Hcv.
      eapply ok_node_mono; [ | exact Hw ]. lia. }
    apply (ext_trans _ _ _ _ He). apply (ext_trans _ _ _ _ H2).
    apply IH; [ lia | ].
    intros first' w' Hin. destruct (Hrest _ _ Hin) as [H3 H4].
    split; [ eapply ok_node_mono; [ | exact H3 ] | intros x Hx;
             eapply ok_node_mono; [ | exact (H4 x Hx) ] ]; lia.
Qed.

Lemma ext_create_non_tupled (a0 : acc) (id : code_id) (ar : arity)
    (params returns : list node) (exn : node) (a : acc) :
  (acc_next a0 <= acc_next a)%positive ->
  (∀ x, x ∈ params -> ok_node a0 a x) ->
  ext a0 a
    (snd (create_unknown_arity_non_tupled_call_witnesses id ar params
            returns exn a))
  ∧ ∀ x,
      x ∈ fst (create_unknown_arity_non_tupled_call_witnesses id ar params
                 returns exn a) ->
      ok_node a0
        (snd (create_unknown_arity_non_tupled_call_witnesses id ar params
                returns exn a)) x.
Proof.
  intros Hle Hp. unfold create_unknown_arity_non_tupled_call_witnesses.
  cbv zeta.
  set (groups := at_least_one_group (group_by_parameter ar params)).
  destruct (ext_fresh_list a0 (length groups) a Hle) as [H1 H1o].
  destruct (fresh_list (length groups) a) as [ws b] eqn:E.
  cbn [fst snd] in *.
  pose proof (ext_next _ _ _ H1) as Hn1.
  assert (H2 : ext a0 b
                 (unknown_arity_witness_deps id returns exn (combine groups ws)
                    b)).
  { apply ext_witness_deps; [ lia | ].
    intros first w Hin. split.
    - exact (H1o w (in_combine_r _ _ _ _ Hin)).
    - intros x Hx. apply in_combine_l in Hin.
      eapply ok_node_mono; [ exact Hn1 | ].
      apply Hp. exact (group_in ar params first x Hin Hx). }
  split; [ exact (ext_trans _ _ _ _ H1 H2) | ].
  intros x Hx. eapply ok_node_mono; [ exact (ext_next _ _ _ H2) | ].
  exact (H1o x Hx).
Qed.

Lemma ext_create_unknown (a0 : acc) (id : code_id) (is_tupled : bool)
    (ar : arity) (params returns : list node) (exn : node) (a : acc) :
  (acc_next a0 <= acc_next a)%positive ->
  (∀ x, x ∈ params -> ok_node a0 a x) ->
  ext a0 a
    (snd (create_unknown_arity_call_witnesses id is_tupled ar params
            returns exn a))
  ∧ ∀ x,
      x ∈ fst (create_unknown_arity_call_witnesses id is_tupled ar params
                 returns exn a) ->
      ok_node a0
        (snd (create_unknown_arity_call_witnesses id is_tupled ar params
                returns exn a)) x.
Proof.
  intros Hle Hp. unfold create_unknown_arity_call_witnesses.
  destruct is_tupled;
    [ apply ext_create_tupled | apply ext_create_non_tupled ]; assumption.
Qed.


(* ------------------------------------------------------------------------ *)
(* The step that makes a code dep                                           *)
(* ------------------------------------------------------------------------ *)

Lemma combine_seq_nth {A : Type} (l : list A) (n i : nat) (x : A) :
  (i, x) ∈ combine (seq n (length l)) l ->
  (n <= i)%nat ∧ nth_error l (i - n) = Some x.
Proof.
  revert n. induction l as [ | y l IH]; intros n H; cbn in H; [ destruct H | ].
  destruct H as [Heq | H].
  - injection Heq as <- <-. rewrite Nat.sub_diag. auto.
  - destruct (IH (S n) H) as [Hle Hx]. split; [ lia | ].
    replace (i - n) with (S (i - S n)) by lia. exact Hx.
Qed.

Lemma indexed_nth {A : Type} (l : list A) (i : nat) (x : A) :
  (i, x) ∈ indexed l -> nth_error l i = Some x.
Proof.
  intros H. destruct (combine_seq_nth l 0 i x H) as [_ Hx].
  rewrite Nat.sub_0_r in Hx. exact Hx.
Qed.

(* [prepare_code] makes the code dep after its witnesses of unknown arity,
   whose facts put nothing into its witness of known arity: they are
   steps from the accumulator [a0] that has the code dep already. *)
Lemma tstep_prepare_code (env : traverse_env) (id : code_id) (c : code)
    (a : acc) :
  tstep env a (prepare_code id c a).
Proof.
  intros Hinv _. unfold prepare_code.
  destruct (ext_fresh_list a (length (code_result_kinds c)) a
              (Pos.le_refl _)) as [H1 H1o].
  destruct (fresh_list _ a) as [returns b1] eqn:E1. cbn [fst snd] in *.
  cbv beta iota.
  fresh_step a b1 exn b2 E2 Hexn. cbv beta iota.
  fresh_step a b2 mc b3 E3 Hmc. cbv beta iota.
  destruct (ext_fresh_list a (length (code_params_kinds c)) b3)
    as [H4 H4o]; [ nexts; lia | ].
  destruct (fresh_list _ b3) as [params b4] eqn:E4. cbn [fst snd] in *.
  cbv beta iota.
  assert (H04 : ext a a b4).
  { apply (ext_trans _ _ _ _ H1). apply (ext_trans _ _ _ _ He).
    apply (ext_trans _ _ _ _ He0). exact H4. }
  destruct (ext_close a b4 Hinv H04) as [Hinv4 Hg4].
  assert (Hok4 : ∀ x, ok_node a b4 x -> good b4 x)
    by (intros x; exact (ok_close a b4 x Hinv H04)).
  assert (Hp4 : ∀ x, x ∈ params -> good b4 x)
    by (intros x Hx; exact (Hok4 x (H4o x Hx))).
  assert (Hr4 : ∀ x, x ∈ returns -> good b4 x).
  { intros x Hx. apply Hok4. pose proof (H1o x Hx). mono. }
  assert (He4 : good b4 exn) by (apply Hok4; mono).
  assert (Hm4 : good b4 mc) by (apply Hok4; mono).
  unfold create_known_arity_call_witness.
  destruct (fresh b4) as [w b4'] eqn:E5.
  destruct (fresh_spec a b4 w b4' E5) as (Hw & Hn5 & _ & _);
    [ nexts; lia | ].
  assert (Hd5 : acc_facts b4' = acc_facts b4
                ∧ acc_code_deps b4' = acc_code_deps b4
                ∧ acc_apply_deps b4' = acc_apply_deps b4
                ∧ acc_closure_deps b4' = acc_closure_deps b4).
  { unfold fresh in E5. injection E5 as Ew Eb. subst b4'. cbn. auto. }
  destruct Hd5 as (Hf5 & Hcd5 & Hap5 & Hcl5).
  cbv beta iota zeta.
  match goal with
  | |- context [add_facts ?L b4'] =>
    set (Lk := L); set (b5 := add_facts Lk b4')
  end.
  assert (Hlt4 : ∀ x, good b4 x -> x <> w).
  { intros x [Hlt _] Hx. rewrite Hx in Hlt. specialize (Hlt _ Hw). lia. }
  (* The facts of the witness of known arity. *)
  assert (HLk : ∀ fct x, fct ∈ Lk -> x ∈ into_nodes fct ->
             (x = w ∧ ∀ cd, kw cd = w -> cd_return cd = returns ->
                known_fact id cd fct)
             ∨ x ∈ params).
  { intros fct x H Hx. unfold Lk in H.
    apply in_app_or in H as [H | H]; [ | apply in_app_or in H as [H | H] ].
    - apply in_map_iff in H as [[i p] [<- Hip]]. cbn [into_nodes] in Hx.
      destruct Hx as [<- | [<- | []]].
      + left. split; [ reflexivity | ]. intros cd Hcd _. cbn [known_fact].
        rewrite Hcd. exact (Hlt4 p (Hp4 p (indexed_in _ _ _ Hip))).
      + right. exact (indexed_in _ _ _ Hip).
    - apply in_map_iff in H as [[i r] [<- Hir]]. cbn [into_nodes] in Hx.
      destruct Hx as [<- | []]. left. split; [ reflexivity | ].
      intros cd _ Hcd. cbn [known_fact]. rewrite Hcd.
      exact (indexed_nth _ _ _ Hir).
    - destruct H as [<- | [<- | []]]; cbn [into_nodes] in Hx;
        destruct Hx as [<- | []]; left; (split; [ reflexivity | ]);
        intros cd _ _; cbn [known_fact]; [ exact I | reflexivity ]. }
  assert (Hcd5' : acc_code_deps b5 = acc_code_deps b4)
    by (unfold b5; cbn [add_facts acc_code_deps]; exact Hcd5).
  assert (Hap5' : acc_apply_deps b5 = acc_apply_deps b4)
    by (unfold b5; cbn [add_facts acc_apply_deps]; exact Hap5).
  assert (Hcl5' : acc_closure_deps b5 = acc_closure_deps b4)
    by (unfold b5; cbn [add_facts acc_closure_deps]; exact Hcl5).
  assert (Hf5' : acc_facts b5 = Lk ++ acc_facts b4)
    by (unfold b5; cbn [add_facts acc_facts]; rewrite Hf5; reflexivity).
  assert (Hn5' : acc_next b5 = Pos.succ (acc_next b4))
    by (unfold b5; cbn [add_facts acc_next]; exact Hn5).
  clearbody b5.
  (* The accumulator with the code dep. *)
  set (cd0 := {| cd_params := params; cd_my_closure := mc;
                 cd_return := returns; cd_exn := exn;
                 cd_known_arity_call_witness := w;
                 cd_unknown_arity_call_witnesses := [] |}).
  set (a0 := add_code_dep id cd0 b5).
  assert (Hk0 : ∀ x, x = w ∨ kwit b4 x -> kwit a0 x).
  { intros x [-> | (id' & cd' & Hin & ->)].
    - exists id, cd0. split; [ left; reflexivity | reflexivity ].
    - exists id', cd'. split; [ | reflexivity ]. right.
      rewrite Hcd5'. exact Hin. }
  assert (Hna0 : acc_next a0 = acc_next b5) by reflexivity.
  assert (Hgood0 : ∀ x, good b4 x -> good a0 x).
  { intros x Hx. pose proof (Hlt4 x Hx) as Hne. destruct Hx as [Hlt Hnk].
    split.
    - intros k Hk. specialize (Hlt k Hk). rewrite Hna0, Hn5'. lia.
    - intros (id' & cd' & Hin & ->). cbn [a0 add_code_dep acc_code_deps]
        in Hin.
      destruct Hin as [Heq | Hin];
        [ injection Heq as <- <-; exact (Hne eq_refl) | ].
      apply Hnk. exists id', cd'. split; [ rewrite <- Hcd5'; exact Hin
                                       | reflexivity ]. }
  destruct (ext_create_unknown a0 id (code_is_tupled c) (code_params_arity c)
              params returns exn b5) as [H6 _];
    [ rewrite Hna0; lia
    | intros x Hx; apply ok_good; exact (Hgood0 x (Hp4 x Hx))
    | ].
  destruct (create_unknown_arity_call_witnesses _ _ _ _ _ _ b5)
    as [unknown b6] eqn:E6.
  cbn [fst snd] in *. cbv beta iota zeta.
  set (b7 := add_fact (Fact_any_source (Node_code_id id)) b6).
  assert (H67 : ext a0 b5 b7).
  { apply (ext_trans _ _ _ _ H6).
    apply ext_add_fact; [ intros y [<- | []]; ok_tac
                        | intros ? ? ?; discriminate ]. }
  clearbody b7.
  destruct H67 as [Hn67 (L & Hf67 & Hi67 & Hw67) Hc67 Hap67 Hcl67].
  match goal with
  | |- context [add_code_dep id ?d b7] => set (cd := d)
  end.
  assert (Hkw : kw cd = w) by reflexivity.
  assert (Hret : cd_return cd = returns) by reflexivity.
  assert (Hkf : ∀ x, kwit (add_code_dep id cd b7) x -> x = w ∨ kwit b4 x).
  { intros x (id' & cd' & Hin & ->). cbn [add_code_dep acc_code_deps] in Hin.
    destruct Hin as [Heq | Hin]; [ injection Heq as <- <-; left; exact Hkw | ].
    right. exists id', cd'. split; [ | reflexivity ].
    rewrite Hc67, Hcd5' in Hin. exact Hin. }
  assert (Hnb : (acc_next b4 < acc_next b7)%positive)
    by (rewrite Hn5' in Hn67; lia).
  assert (Hgoodf : ∀ x, good b4 x -> good (add_code_dep id cd b7) x).
  { intros x Hx. pose proof (Hlt4 x Hx) as Hne. destruct Hx as [Hlt Hnk].
    split.
    - intros k Hk. specialize (Hlt k Hk). cbn [add_code_dep acc_next]. lia.
    - intros Hk. destruct (Hkf x Hk) as [Hk' | Hk'];
        [ exact (Hne Hk') | exact (Hnk Hk') ]. }
  assert (Hokf : ∀ x, ok_node a0 b7 x -> good (add_code_dep id cd b7) x).
  { intros x [Hx | [[Hlt Hnk] | (k & -> & Hk1 & Hk2)]].
    - split; [ intros k -> ; destruct (Hx k eq_refl) | ].
      intros Hk. destruct (Hkf x Hk) as [-> | Hk'];
        [ exact (Hx _ Hw) | ].
      destruct (kwit_made b4 x Hinv4 Hk') as (k & -> & _).
      exact (Hx k eq_refl).
    - split; [ intros k Hk; specialize (Hlt k Hk); cbn [add_code_dep acc_next];
               lia | ].
      intros Hk. exact (Hnk (Hk0 x (Hkf x Hk))).
    - split; [ intros k' Hk'; injection Hk' as <-; exact Hk2 | ].
      intros Hk. rewrite Hna0, Hn5' in Hk1.
      destruct (Hkf _ Hk) as [Hk' | Hk'].
      + rewrite Hw in Hk'. injection Hk' as ->. lia.
      + destruct (kwit_made b4 _ Hinv4 Hk') as (k' & Hk'' & Hlt).
        injection Hk'' as <-. lia. }
  assert (Hff : ∀ fct, fct ∈ acc_facts (add_code_dep id cd b7) ->
             fct ∈ L ∨ fct ∈ Lk ∨ fct ∈ acc_facts b4).
  { intros fct Hin. cbn [add_code_dep acc_facts] in Hin.
    rewrite Hf67, Hf5' in Hin.
    apply in_app_or in Hin as [Hin | Hin]; [ left; exact Hin | right ].
    apply in_app_or in Hin as [Hin | Hin]; [ left | right ]; exact Hin. }
  split; [ constructor | ].
  - intros fct x Hin Hx. destruct (Hff fct Hin) as [H | [H | H]].
    + exact (proj1 (Hokf x (Hi67 _ _ H Hx))).
    + destruct (HLk fct x H Hx) as [[-> _] | Hpx].
      * intros k Hk. rewrite Hw in Hk. injection Hk as <-.
        cbn [add_code_dep acc_next]. exact Hnb.
      * exact (proj1 (Hgoodf x (Hp4 x Hpx))).
    + intros k Hk. specialize (ki_into b4 Hinv4 _ _ H Hx k Hk).
      cbn [add_code_dep acc_next]. lia.
  - intros id' cd' Hin. cbn [add_code_dep acc_code_deps] in Hin.
    destruct Hin as [Heq | Hin].
    + injection Heq as <- <-. exists (acc_next b4).
      split; [ rewrite Hkw; exact Hw | cbn [add_code_dep acc_next]; exact Hnb ].
    + rewrite Hc67, Hcd5' in Hin.
      destruct (ki_made b4 Hinv4 _ _ Hin) as (k & Hk & Hlt).
      exists k. split; [ exact Hk | cbn [add_code_dep acc_next]; lia ].
  - intros id' cd' fct Hin Hfct Hx.
    assert (Hkc : kwit (add_code_dep id cd b7) (kw cd'))
      by (exists id', cd'; split; [ exact Hin | reflexivity ]).
    cbn [add_code_dep acc_code_deps] in Hin.
    destruct (Hff fct Hfct) as [H | [H | H]].
    + exfalso. destruct (Hokf _ (Hi67 _ _ H Hx)) as [_ Hnk]. exact (Hnk Hkc).
    + destruct Hin as [Heq | Hin].
      * injection Heq as <- <-. destruct (HLk fct _ H Hx) as [[_ Hk] | Hp].
        -- exact (Hk cd Hkw Hret).
        -- exfalso. exact (Hlt4 _ (Hp4 _ Hp) Hkw).
      * rewrite Hc67, Hcd5' in Hin. exfalso.
        destruct (HLk fct _ H Hx) as [[Hxw _] | Hp].
        -- destruct (ki_made b4 Hinv4 _ _ Hin) as (k & Hk & Hlt).
           rewrite Hxw, Hw in Hk. injection Hk as Hk. lia.
        -- destruct (Hp4 _ Hp) as [_ Hnk]. apply Hnk.
           exists id', cd'. split; [ exact Hin | reflexivity ].
    + destruct Hin as [Heq | Hin].
      * injection Heq as <- <-. exfalso.
        pose proof (ki_into b4 Hinv4 _ _ H Hx) as Hlt.
        rewrite Hkw in Hlt. specialize (Hlt _ Hw). lia.
      * rewrite Hc67, Hcd5' in Hin. exact (ki_known b4 Hinv4 _ _ _ Hin H Hx).
  - intros b from Hin. destruct (Hff _ Hin) as [H | [H | H]].
    + exact (Hw67 _ _ H).
    + unfold Lk in H.
      apply in_app_or in H as [H | H]; [ | apply in_app_or in H as [H | H] ].
      * apply in_map_iff in H as [[i p] [Heq _]]. discriminate Heq.
      * apply in_map_iff in H as [[i r] [Heq _]]. discriminate Heq.
      * destruct H as [H | [H | []]]; discriminate H.
    + exact (ki_constructor b4 Hinv4 _ _ H).
  - intros id' cd' x Hin Hx. cbn [add_code_dep acc_code_deps] in Hin.
    destruct Hin as [Heq | Hin].
    + injection Heq as <- <-. unfold cd, code_dep_nodes in Hx.
      cbn [cd_my_closure cd_exn cd_return] in Hx.
      destruct Hx as [<- | [<- | Hx]]; apply Hgoodf;
        [ exact Hm4 | exact He4 | exact (Hr4 _ Hx) ].
    + rewrite Hc67, Hcd5' in Hin.
      exact (Hgoodf _ (ki_code_deps b4 Hinv4 _ _ _ Hin Hx)).
  - intros d Hin. cbn [add_code_dep acc_apply_deps] in Hin.
    destruct (Hap67 _ Hin) as [Hin' | H].
    + rewrite Hap5' in Hin'. exact (Hgoodf _ (ki_apply_deps b4 Hinv4 _ Hin')).
    + exact (Hokf _ H).
  - intros d Hin. cbn [add_code_dep acc_closure_deps] in Hin.
    destruct (Hcl67 _ Hin) as [Hin' | H].
    + rewrite Hcl5' in Hin'.
      exact (Hgoodf _ (ki_closure_deps b4 Hinv4 _ Hin')).
    + exact (Hokf _ H).
  - pose proof (ext_next _ _ _ H04) as Ha4. split.
    + cbn [add_code_dep acc_next]. lia.
    + intros x Hk. destruct (Hkf x Hk) as [-> | Hk'].
      * right. exists (acc_next b4). split; [ exact Hw | lia ].
      * exact (proj2 Hg4 x Hk').
Qed.

Lemma tstep_prepare_binding (env : traverse_env) (binding : static_binding)
    (a : acc) :
  tstep env a (prepare_binding binding a).
Proof.
  destruct binding; cbn [prepare_binding];
    [ apply tstep_prepare_code | apply tstep_refl | apply tstep_refl ].
Qed.

(* ------------------------------------------------------------------------ *)
(* Calls                                                                    *)
(* ------------------------------------------------------------------------ *)

Lemma ext_unknown_apply_deps (a0 : acc) (returns : list node)
    (exn called : node) (aw : list (list simple * node)) (a : acc) :
  (acc_next a0 <= acc_next a)%positive ->
  (∀ x, x ∈ exn :: called :: returns -> ok_node a0 a x) ->
  (∀ first w, (first, w) ∈ aw -> ok_node a0 a w) ->
  ext a0 a (unknown_arity_apply_deps returns exn called aw a).
Proof.
  revert a. induction aw as [ | [first w] rest IH]; intros a Hle Hin Haw;
    cbn [unknown_arity_apply_deps]; [ apply ext_refl | ].
  set (b := add_facts _ a).
  assert (Hbn : acc_next b = acc_next a) by reflexivity.
  assert (H1 : ext a0 a b).
  { apply ext_add_facts; [ | no_known ].
    intros fct x H Hx. into_split H; into_close H Hx; apply Hin;
      [ left | right; left ]; reflexivity. }
  apply (ext_trans _ _ _ _ H1).
  destruct rest as [ | [g nw] rest'].
  - apply ext_add_facts; [ | no_known ].
    intros fct x H Hx. into_split H; into_close H Hx.
    match goal with
    | Hy : (?i, ?r) ∈ indexed returns |- _ =>
      assert (Hr : ok_node a0 a r)
        by (apply Hin; right; right; eapply indexed_in; exact Hy)
    end.
    mono.
  - fresh_step a0 b v c E Hv. cbv zeta.
    set (d := add_facts _ c).
    assert (Hdn : acc_next d = acc_next c) by reflexivity.
    assert (H2 : ext a0 c d).
    { pose proof (Haw g nw (or_intror (or_introl eq_refl))) as Hnw.
      apply ext_add_facts; [ | no_known ].
      intros fct x H Hx. into_split H; into_close H Hx; [ exact Ho | ].
      mono. }
    apply (ext_trans _ _ _ _ He). apply (ext_trans _ _ _ _ H2).
    apply IH; [ lia | | ].
    + intros x Hx. pose proof (Hin x Hx). mono.
    + intros first' w' Hw'. pose proof (Haw _ _ (or_intror Hw')). mono.
Qed.

Lemma ext_make_unknown (a0 : acc) (env : traverse_env) (ap : apply_expr)
    (returns : list node) (exn : node) (a : acc) :
  (acc_next a0 <= acc_next a)%positive ->
  (∀ x, x ∈ exn :: returns -> ok_node a0 a x) ->
  ext a0 a (snd (make_unknown_arity_apply_widget env ap returns exn a))
  ∧ ok_node a0 (snd (make_unknown_arity_apply_widget env ap returns exn a))
      (fst (make_unknown_arity_apply_widget env ap returns exn a)).
Proof.
  intros Hle Hin. unfold make_unknown_arity_apply_widget.
  fresh_step a0 a called b E Hc. cbv beta iota zeta.
  set (b1 := add_fact _ b).
  assert (H1 : ext a0 b b1).
  { apply ext_add_fact; [ intros x [] | intros ? ? ?; discriminate ]. }
  set (groups := at_least_one_group _).
  destruct (ext_fresh_list a0 (length groups) b1) as [H2 H2o];
    [ nexts; lia | ].
  destruct (fresh_list (length groups) b1) as [ws b2] eqn:E2.
  cbn [fst snd] in *. cbv beta iota zeta.
  destruct (note_simples_fields (ap_args ap) b2) as (N1 & _).
  pose proof (ext_note_simples a0 (ap_args ap) b2) as H3.
  set (b3 := note_simples (ap_args ap) b2) in *.
  assert (H4 : ext a0 b3
                 (unknown_arity_apply_deps returns exn called
                    (combine groups ws) b3)).
  { apply ext_unknown_apply_deps; [ nexts; lia | | ].
    - intros x [<- | [<- | Hx]].
      + pose proof (Hin exn (or_introl eq_refl)). mono.
      + mono.
      + pose proof (Hin x (or_intror Hx)). mono.
    - intros first w Hw. pose proof (H2o w (in_combine_r _ _ _ _ Hw)).
      mono. }
  set (b4 := unknown_arity_apply_deps _ _ _ _ b3) in *.
  fresh_step a0 b4 an b5 E5 Han.
  assert (H05 : ext a0 a b5).
  { apply (ext_trans _ _ _ _ He). apply (ext_trans _ _ _ _ H1).
    apply (ext_trans _ _ _ _ H2). apply (ext_trans _ _ _ _ H3).
    apply (ext_trans _ _ _ _ H4). exact He0. }
  destruct ws as [ | first rest]; cbn [fst snd].
  - split; [ exact H05 | exact Ho0 ].
  - pose proof (H2o first (or_introl eq_refl)) as Hf.
    assert (H6 : ext a0 b5 (cond_alias env an first b5)).
    { apply ext_cond_alias. mono. }
    split; [ exact (ext_trans _ _ _ _ H05 H6) | ].
    unfold cond_alias. destruct (env_current_code_id env); cbn;
      exact Ho0.
Qed.


Lemma ext_make_known (a0 : acc) (env : traverse_env) (ap : apply_expr)
    (returns : list node) (exn : node) (a : acc) :
  (acc_next a0 <= acc_next a)%positive ->
  (∀ x, x ∈ exn :: returns -> ok_node a0 a x) ->
  ext a0 a (snd (make_known_arity_apply_widget env ap returns exn a))
  ∧ ok_node a0 (snd (make_known_arity_apply_widget env ap returns exn a))
      (fst (make_known_arity_apply_widget env ap returns exn a)).
Proof.
  intros Hle Hin. unfold make_known_arity_apply_widget.
  cbn [fresh]. cbv beta iota zeta. cbn [fst snd acc_next].
  destruct (note_simples_fields (ap_args ap)
              {| acc_next := Pos.succ (Pos.succ (Pos.succ (acc_next a)));
                 acc_facts := acc_facts a;
                 acc_code_deps := acc_code_deps a;
                 acc_apply_deps := acc_apply_deps a;
                 acc_closure_deps := acc_closure_deps a;
                 acc_symbols := acc_symbols a |})
    as (N1 & N2 & N3 & N4 & N5).
  cbn [acc_next acc_facts acc_code_deps acc_apply_deps acc_closure_deps]
    in N1, N2, N3, N4, N5.
  set (b := note_simples _ _) in *. clearbody b.
  set (c := add_facts _ b).
  assert (Hext : ext a0 b (cond_alias env
                             (Node_made (Pos.succ (Pos.succ (acc_next a))))
                             (Node_made (acc_next a)) c)).
  { apply (ext_trans _ _ c).
    - apply ext_add_facts; [ | no_known ].
      intros fct x H Hx. into_split H; into_close H Hx.
      + eapply ok_node_mono; [ | apply Hin; right; eapply indexed_in;
                                 eassumption ].
        rewrite N1. lia.
      + eapply ok_node_mono; [ | apply Hin; left; reflexivity ].
        rewrite N1. lia.
      + apply ok_made; try rewrite N1; lia.
    - apply ext_cond_alias. apply ok_made; cbn; try rewrite N1; lia. }
  split; [ apply (ext_trans _ _ b); [ | exact Hext ];
           apply ext_same; auto; rewrite N1; lia | ].
  apply ok_made; [ lia | ].
  unfold cond_alias; destruct (env_current_code_id env); cbn; rewrite N1;
    lia.
Qed.

Lemma ext_add_apply (a0 : acc) (env : traverse_env) (id : code_id)
    (callee call_widget : node) (only_if_closure_any_source : bool)
    (a : acc) :
  (acc_next a0 <= acc_next a)%positive -> ok_node a0 a call_widget ->
  ext a0 a (add_apply env id callee call_widget only_if_closure_any_source a).
Proof.
  intros Hle Hcw. unfold add_apply.
  destruct only_if_closure_any_source; [ | apply ext_push_apply_dep; auto ].
  fresh_step a0 a cif b E Hcif. cbv beta iota.
  fresh_step a0 b wif d E2 Hwif. cbv beta iota.
  apply (ext_trans _ _ _ _ He). apply (ext_trans _ _ _ _ He0).
  match goal with
  | |- ext _ _ (push_apply_dep _ ?b) => apply (ext_trans _ _ b)
  end.
  - apply ext_add_facts; [ | no_known ].
    intros fct x H Hx. into_split H; into_close H Hx; mono.
  - apply ext_push_apply_dep. cbn [ad_apply_call_witness]. mono.
Qed.

Lemma ext_call_kind (a0 : acc) (env : traverse_env) (ap : apply_expr)
    (exn_arg : node) (return_args : list node) (a : acc) :
  (acc_next a0 <= acc_next a)%positive ->
  (∀ x, x ∈ exn_arg :: return_args -> ok_node a0 a x) ->
  ext a0 a (traverse_call_kind env ap exn_arg return_args a).
Proof.
  intros Hle Hin. unfold traverse_call_kind. cbv zeta.
  pose proof (ext_note_simple a0 (ap_callee ap) a) as H1.
  set (a1 := note_simple (ap_callee ap) a) in *. clearbody a1.
  set (closure := node_of_simple (ap_callee ap)).
  assert (Hin1 : ∀ x, x ∈ exn_arg :: return_args -> ok_node a0 a1 x).
  { intros x Hx. pose proof (Hin x Hx). mono. }
  apply (ext_trans _ _ _ _ H1).
  destruct (ap_call_kind ap) as [id | | code_ids].
  - destruct (ext_make_known a0 env ap return_args exn_arg a1)
      as [Hk Hcw]; [ nexts; lia | exact Hin1 | ].
    destruct (make_known_arity_apply_widget env ap return_args exn_arg a1)
      as [cw a2] eqn:E. cbn [fst snd] in *. cbv beta iota zeta.
    set (a3 := add_fact (Fact_accessor cw known closure) a2).
    assert (H3 : ext a0 a2 a3).
    { apply ext_add_fact; [ intros y [<- | []]; exact Hcw | ].
      intros ? ? ?; discriminate. }
    apply (ext_trans _ _ _ _ Hk). apply (ext_trans _ _ _ _ H3).
    assert (Hcw3 : ok_node a0 a3 cw) by mono.
    clearbody a3.
    destruct (env_should_preserve_direct_calls env);
      [ | destruct (negb (among id (env_unit_code_ids env))) | ];
      try apply ext_refl; apply ext_add_apply; try exact Hcw3; nexts; lia.
  - destruct (ext_make_unknown a0 env ap return_args exn_arg a1)
      as [Hk Hcw]; [ nexts; lia | exact Hin1 | ].
    destruct (make_unknown_arity_apply_widget env ap return_args exn_arg a1)
      as [cw a2] eqn:E. cbn [fst snd] in *. cbv beta iota.
    apply (ext_trans _ _ _ _ Hk).
    apply ext_add_fact; [ intros y [<- | []]; exact Hcw | ].
    intros ? ? ?; discriminate.
  - destruct (ext_make_known a0 env ap return_args exn_arg a1)
      as [Hk Hcw]; [ nexts; lia | exact Hin1 | ].
    destruct (make_known_arity_apply_widget env ap return_args exn_arg a1)
      as [cw a2] eqn:E. cbn [fst snd] in *. cbv beta iota zeta.
    apply (ext_trans _ _ _ _ Hk).
    apply ext_add_fact; [ intros y [<- | []]; exact Hcw | ].
    intros ? ? ?; discriminate.
Qed.

Lemma ext_traverse_apply (a0 : acc) (env : traverse_env) (ap : apply_expr)
    (a : acc) :
  env_good env a0 -> (acc_next a0 <= acc_next a)%positive ->
  ext a0 a (traverse_apply env ap a).
Proof.
  intros Henv Hle. unfold traverse_apply.
  destruct (ext_find_cont a0 env (ap_continuation ap)
              (length (unarize (ap_return_arity ap))) a Henv Hle)
    as [H1 H1o].
  destruct (find_cont env (ap_continuation ap) _ a) as [rets b1] eqn:E1.
  cbn [fst snd] in *. cbv beta iota.
  destruct (ext_find_cont a0 env (ap_exn_continuation ap)
              (S (length (ap_exn_extra_args ap))) b1 Henv)
    as [H2 H2o]; [ nexts; lia | ].
  destruct (find_cont env (ap_exn_continuation ap) _ b1) as [eps b2] eqn:E2.
  cbn [fst snd] in *. cbv beta iota.
  apply (ext_trans _ _ _ _ H1). apply (ext_trans _ _ _ _ H2).
  assert (Hpre : ∀ (exn_arg : node) (b3 : acc),
            ext a0 b2 b3 -> ok_node a0 b3 exn_arg ->
            ext a0 b2 (traverse_call_kind env ap exn_arg rets b3)).
  { intros exn_arg b3 H3 Hex. apply (ext_trans _ _ _ _ H3).
    apply ext_call_kind; [ nexts; lia | ].
    intros x [<- | Hx]; [ exact Hex | ].
    pose proof (H1o x Hx). mono. }
  destruct eps as [ | ep extra].
  - destruct (fresh b2) as [x b3] eqn:E3. cbv beta iota.
    destruct (fresh_spec a0 b2 x b3 E3) as (_ & _ & He & Ho);
      [ nexts; lia | ].
    exact (Hpre x b3 He Ho).
  - cbv zeta beta iota.
    pose proof (ext_note_simples a0 (exn_extra_simples ap) b2) as H3.
    destruct (note_simples_fields (exn_extra_simples ap) b2) as (N1 & _).
    set (b3 := note_simples _ b2) in *. clearbody b3.
    apply Hpre.
    + apply (ext_trans _ _ _ _ H3). apply ext_add_facts; [ | no_known ].
      intros fct y H Hy. into_split H; into_close H Hy.
      match goal with
      | Hc : (?p, ?s) ∈ combine extra _ |- _ =>
        pose proof (H2o p (or_intror (in_combine_l _ _ _ _ Hc)))
      end.
      mono.
    + pose proof (H2o ep (or_introl eq_refl)). mono.
Qed.

Lemma ext_apply_cont_deps (a0 : acc) (env : traverse_env) (ac : apply_cont)
    (a : acc) :
  env_good env a0 -> (acc_next a0 <= acc_next a)%positive ->
  ext a0 a (apply_cont_deps env ac a).
Proof.
  intros Henv Hle. unfold apply_cont_deps. cbv zeta.
  destruct (ext_find_cont a0 env (ac_continuation ac) (length (ac_args ac)) a
              Henv Hle) as [H1 H1o].
  destruct (find_cont env (ac_continuation ac) _ a) as [params b1] eqn:E1.
  cbn [fst snd] in *. cbv beta iota zeta.
  pose proof (ext_note_simples a0 (ac_args ac) b1) as H2.
  apply (ext_trans _ _ _ _ H1). apply (ext_trans _ _ _ _ H2).
  apply ext_add_facts; [ | no_known ].
  intros fct y H Hy. into_split H; into_close H Hy.
  match goal with
  | Hc : (?p, ?s) ∈ combine params _ |- _ =>
    pose proof (H1o p (in_combine_l _ _ _ _ Hc))
  end.
  mono.
Qed.

Lemma ext_traverse_switch (a0 : acc) (env : traverse_env) (scrutinee : simple)
    (arms : list (Z * apply_cont)) (a : acc) :
  env_good env a0 -> (acc_next a0 <= acc_next a)%positive ->
  ext a0 a (traverse_switch env scrutinee arms a).
Proof.
  intros Henv Hle. unfold traverse_switch. cbv zeta.
  pose proof (ext_cond_any_usage a0 env scrutinee a) as H1.
  apply (ext_trans _ _ _ _ H1). apply ext_fold.
  intros [i ac] b _ Hb. apply ext_apply_cont_deps; [ exact Henv | ].
  pose proof (ext_next _ _ _ H1). lia.
Qed.

Lemma ext_exn_handler_deps (a0 : acc) (env : traverse_env)
    (h : continuation_handler) (a : acc) :
  ext a0 a (exn_handler_deps env h a).
Proof.
  unfold exn_handler_deps. destruct h as [[ | p ps] md body];
    [ apply ext_refl | ].
  destruct (hm_is_exn_handler md); [ | apply ext_refl ].
  match goal with
  | |- ext _ _ (cond_any_usage _ _ ?b) => apply (ext_trans _ _ b)
  end; [ | apply ext_cond_any_usage ].
  apply ext_add_fact; [ intros y [<- | []]; ok_tac | ].
  intros ? ? ?; discriminate.
Qed.


(* ------------------------------------------------------------------------ *)
(* The traversal                                                            *)
(* ------------------------------------------------------------------------ *)

Lemma good_not_made (a : acc) (x : node) :
  kinv a -> not_made x -> good a x.
Proof.
  intros Hinv Hx. split; [ intros k -> ; destruct (Hx k eq_refl) | ].
  intros Hw. destruct (kwit_made a x Hinv Hw) as (k & -> & _).
  exact (Hx k eq_refl).
Qed.

Lemma tstep_env (env env' : traverse_env) (a a' : acc) :
  (kinv a -> env_good env a -> env_good env' a) ->
  tstep env' a a' -> tstep env a a'.
Proof. intros H H' Hinv Henv. exact (H' Hinv (H Hinv Henv)). Qed.

Lemma env_good_add_cont (env : traverse_env) (k : continuation)
    (params : list node) (a : acc) :
  env_good env a -> (∀ x, x ∈ params -> good a x) ->
  env_good (add_cont env k params) a.
Proof.
  intros Henv Hp k' params' x [Heq | Hin] Hx.
  - injection Heq as <- <-. exact (Hp x Hx).
  - exact (Henv _ _ _ Hin Hx).
Qed.

Lemma handler_params_good (a : acc) (h : continuation_handler) :
  kinv a -> ∀ x, x ∈ handler_params h -> good a x.
Proof.
  intros Hinv x Hx. destruct h as [params md body]. cbn in Hx.
  apply in_map_iff in Hx. destruct Hx as [y [<- _]].
  apply good_not_made; [ exact Hinv | intros k; discriminate ].
Qed.

Lemma env_good_add_rec_conts (env : traverse_env)
    (handlers : list (continuation * continuation_handler)) (a : acc) :
  kinv a -> env_good env a -> env_good (add_rec_conts env handlers) a.
Proof.
  intros Hinv Henv. unfold add_rec_conts.
  induction handlers as [ | [k h] handlers IH]; cbn [fold_right];
    [ exact Henv | ].
  apply env_good_add_cont; [ exact IH | ].
  apply handler_params_good. exact Hinv.
Qed.

Lemma tstep_traverse :
  (∀ e env a, tstep env a (traverse env e a))
  ∧ (∀ h env a, tstep env a (traverse_handler env h a)).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - intros x kind d body IH env a. cbn [traverse].
    apply (tstep_trans _ _ (traverse_named env x d a)); [ | apply IH ].
    apply tstep_ext. intros _ _. apply ext_traverse_named. lia.
  - intros bindings body IHc IH env a. cbn [traverse]. cbv zeta.
    eapply tstep_trans; [ | apply IH ].
    eapply tstep_trans; [ | apply tstep_fold ].
    + eapply tstep_trans; [ | apply tstep_fold ].
      * apply tstep_fold. intros b a' _. apply tstep_prepare_binding.
      * intros b a' _. apply tstep_ext. intros _ _.
        apply ext_traverse_static_binding.
    + intros [id c | id | s const] a' Hin; cbn [traverse_binding];
        [ | apply tstep_refl | apply tstep_refl ].
      destruct c as [params mc ret exn md cbody]. cbn [traverse_code].
      destruct (lookup id (acc_code_deps a')) as [cd | ] eqn:Ecd;
        [ | apply tstep_refl ].
      set (b := add_facts _ a').
      apply (tstep_trans _ _ b).
      * apply tstep_ext. intros _ _. apply ext_add_facts; [ | no_known ].
        intros fct y H Hy. into_split H; into_close H Hy; ok_tac.
      * apply (tstep_env _ (code_env env id ret exn cd));
          [ | apply (IHc id (Code params mc ret exn md cbody) Hin) ].
        intros Hinvb _ k ps y Hk Hy.
        assert (Hin' : (id, cd) ∈ acc_code_deps b)
          by exact (FlambdaMapProofs.lookup_in _ _ _ Ecd).
        cbn [code_env env_conts] in Hk.
        destruct Hk as [Hk | [Hk | []]]; injection Hk as <- <-;
          apply (ki_code_deps b Hinvb id cd y Hin');
          cbn [code_dep_nodes];
          [ right; left; destruct Hy as [Hy | []]; exact Hy
          | right; right; exact Hy ].
  - intros k h body IHh IH env a. cbn [traverse]. cbv zeta.
    eapply tstep_trans; [ apply IHh | ].
    eapply tstep_trans; [ apply tstep_ext; intros _ _;
                          apply ext_exn_handler_deps | ].
    apply (tstep_env _ (add_cont env k (handler_params h))); [ | apply IH ].
    intros Hinv Henv. apply env_good_add_cont; [ exact Henv | ].
    apply handler_params_good. exact Hinv.
  - intros handlers body IHh IH env a. cbn [traverse]. cbv zeta.
    apply (tstep_env _ (add_rec_conts env handlers));
      [ intros Hinv Henv; apply env_good_add_rec_conts; assumption | ].
    eapply tstep_trans; [ | apply IH ].
    apply tstep_fold. intros [k h] a' Hin. apply (IHh k h Hin).
  - intros ap env a. apply tstep_ext. intros _ Henv.
    apply ext_traverse_apply; [ exact Henv | lia ].
  - intros ac env a. apply tstep_ext. intros _ Henv.
    apply ext_apply_cont_deps; [ exact Henv | lia ].
  - intros scrutinee arms env a. apply tstep_ext. intros _ Henv.
    apply ext_traverse_switch; [ exact Henv | lia ].
  - intros env a. apply tstep_refl.
  - intros params md handler IH env a. cbn [traverse_handler]. apply IH.
Qed.

(* ------------------------------------------------------------------------ *)
(* The unit                                                                 *)
(* ------------------------------------------------------------------------ *)

(* A node that the facts of a step from [a] may put something into is not
   a witness of known arity of [a]. *)
Lemma ok_not_kwit (a a' : acc) (x : node) :
  kinv a -> ok_node a a' x -> ~ kwit a x.
Proof.
  intros Hinv Hx Hk. destruct (kwit_made a x Hinv Hk) as (k & -> & Hlt).
  destruct Hx as [Hx | [[_ Hnk] | (k' & Hk' & H1 & _)]].
  - exact (Hx k eq_refl).
  - exact (Hnk Hk).
  - injection Hk' as <-. lia.
Qed.

(* [add_closure_dep], which makes the constructors of the field of known
   arity: of the witness of a code dep, or of a node with any source. *)
Lemma closure_dep_step (a0 : acc) (cds : finite_map code_dep)
    (d : closure_dep) (b : acc) :
  (acc_next a0 <= acc_next b)%positive ->
  ok_node a0 b (cld_let_bound_name_of_the_closure d) ->
  (∀ id cd, lookup id cds = Some cd -> ok_node a0 b (cd_my_closure cd)) ->
  acc_code_deps (add_closure_dep cds d b) = acc_code_deps b
  ∧ (acc_next b <= acc_next (add_closure_dep cds d b))%positive
  ∧ ∃ L, acc_facts (add_closure_dep cds d b) = L ++ acc_facts b
    ∧ (∀ fct x, fct ∈ L -> x ∈ into_nodes fct ->
         ok_node a0 (add_closure_dep cds d b) x)
    ∧ (∀ c from, Fact_constructor c known from ∈ L ->
         (∃ id cd, lookup id cds = Some cd ∧ from = kw cd)
         ∨ Fact_any_source from ∈ L).
Proof.
  intros Hle Hc Hmc. unfold add_closure_dep.
  destruct (lookup (cld_closure_code_id d) cds) as [cd | ] eqn:E.
  - cbn [add_facts acc_code_deps acc_next acc_facts].
    split; [ reflexivity | split; [ lia | ] ].
    eexists. split; [ reflexivity | split ].
    + intros fct x H Hx. into_split H; into_close H Hx;
        try exact Hc; exact (Hmc _ _ E).
    + intros c' from H. left. exists (cld_closure_code_id d), cd.
      split; [ exact E | ].
      into_split H; try discriminate H; injection H as _ <-; reflexivity.
  - destruct (fresh b) as [w b'] eqn:Ef. cbv beta iota.
    destruct (fresh_spec a0 b w b' Ef Hle) as (Hw & Hn & He & Ho).
    pose proof (ext_code_deps _ _ _ He) as Hcd.
    assert (Hf : acc_facts b' = acc_facts b).
    { unfold fresh in Ef. injection Ef as _ <-. reflexivity. }
    cbn [add_facts acc_code_deps acc_next acc_facts].
    split; [ exact Hcd | split; [ lia | ] ].
    eexists. split; [ rewrite Hf; reflexivity | split ].
    + intros fct x H Hx. into_split H; into_close H Hx; try exact Ho;
        eapply ok_node_mono; [ | exact Hc | | exact Hc ]; cbn; lia.
    + intros c' from H. right.
      into_split H; try discriminate H; injection H as _ <-; left;
        reflexivity.
Qed.

Lemma closure_deps_fold (a0 : acc) (cds : finite_map code_dep)
    (ds : list closure_dep) (b : acc) :
  (acc_next a0 <= acc_next b)%positive ->
  (∀ d, d ∈ ds -> ok_node a0 b (cld_let_bound_name_of_the_closure d)) ->
  (∀ id cd, lookup id cds = Some cd -> ok_node a0 b (cd_my_closure cd)) ->
  let b' := List.fold_left (fun a d => add_closure_dep cds d a) ds b in
  acc_code_deps b' = acc_code_deps b
  ∧ (acc_next b <= acc_next b')%positive
  ∧ ∃ L, acc_facts b' = L ++ acc_facts b
    ∧ (∀ fct x, fct ∈ L -> x ∈ into_nodes fct -> ok_node a0 b' x)
    ∧ (∀ c from, Fact_constructor c known from ∈ L ->
         (∃ id cd, lookup id cds = Some cd ∧ from = kw cd)
         ∨ Fact_any_source from ∈ L).
Proof.
  revert b. induction ds as [ | d ds IH]; intros b Hle Hds Hmc;
    cbn [List.fold_left].
  - split; [ reflexivity | split; [ lia | ] ].
    exists []. split; [ reflexivity | split ];
      [ intros fct x [] | intros c from [] ].
  - destruct (closure_dep_step a0 cds d b Hle (Hds d (or_introl eq_refl))
                Hmc) as (Hc1 & Hn1 & L1 & Hf1 & Hi1 & Hk1).
    set (b1 := add_closure_dep cds d b) in *. clearbody b1.
    destruct (IH b1) as (Hc2 & Hn2 & L2 & Hf2 & Hi2 & Hk2); [ lia | | | ].
    { intros d' Hd'. eapply ok_node_mono; [ exact Hn1 | ].
      exact (Hds d' (or_intror Hd')). }
    { intros id cd E. eapply ok_node_mono; [ exact Hn1 | ].
      exact (Hmc id cd E). }
    split; [ congruence | split; [ lia | ] ].
    exists (L2 ++ L1). split; [ rewrite Hf2, Hf1, app_assoc; reflexivity | ].
    split.
    + intros fct x H Hx. apply in_app_or in H as [H | H];
        [ exact (Hi2 _ _ H Hx) | ].
      eapply ok_node_mono; [ exact Hn2 | exact (Hi1 _ _ H Hx) ].
    + intros c from H. apply in_app_or in H as [H | H].
      * destruct (Hk2 _ _ H) as [H' | H']; [ left; exact H' | ].
        right. apply in_or_app. left. exact H'.
      * destruct (Hk1 _ _ H) as [H' | H']; [ left; exact H' | ].
        right. apply in_or_app. right. exact H'.
Qed.

(* What the delayed deps keep of the invariant: the facts that they add
   put nothing into a witness of known arity, and their constructors of
   the field of known arity hold the witness of a code dep or a node with
   any source. *)
Lemma resolve_known (a : acc) (unit_symbols : list symbol) :
  kinv a ->
  acc_code_deps (resolve_delayed_deps unit_symbols a) = acc_code_deps a
  ∧ (∀ id cd fct,
       (id, cd) ∈ acc_code_deps a ->
       fct ∈ acc_facts (resolve_delayed_deps unit_symbols a) ->
       kw cd ∈ into_nodes fct -> known_fact id cd fct)
  ∧ (∀ b from,
       Fact_constructor b known from
       ∈ acc_facts (resolve_delayed_deps unit_symbols a) ->
       (∃ id cd, (id, cd) ∈ acc_code_deps a ∧ from = kw cd)
       ∨ Fact_any_source from
         ∈ acc_facts (resolve_delayed_deps unit_symbols a)).
Proof.
  intros Hinv. unfold resolve_delayed_deps. cbv zeta.
  assert (Hcd : ∀ id cd x, lookup id (acc_code_deps a) = Some cd ->
                  x ∈ code_dep_nodes cd -> ok_node a a x).
  { intros id cd x E Hx. apply ok_good.
    exact (ki_code_deps a Hinv id cd x (FlambdaMapProofs.lookup_in _ _ _ E)
             Hx). }
  set (b1 := add_facts _ a).
  assert (H1 : ext a a b1).
  { apply ext_add_facts.
    - intros fct y H Hy. apply in_flat_map in H. destruct H as [d [Hd H]].
      pose proof (ok_good a a _ (ki_apply_deps a Hinv d Hd)) as Hw.
      unfold apply_dep_facts, alias_for_caller in H.
      destruct (lookup (ad_apply_code_id d) (acc_code_deps a)) as [cd | ]
        eqn:E;
        destruct (ad_function_containing_apply_expr d);
        into_split H; into_close H Hy; try exact Hw; try ok_tac;
        apply (Hcd _ cd _ E); left; reflexivity.
    - intros w n H. apply in_flat_map in H. destruct H as [d [_ H]].
      unfold apply_dep_facts, alias_for_caller in H.
      destruct (lookup _ _); destruct (ad_function_containing_apply_expr d);
        into_split H; discriminate. }
  clearbody b1.
  destruct H1 as [Hn1 (L1 & Hf1 & Hi1 & Hw1) Hc1 Hap1 Hcl1].
  destruct (closure_deps_fold a (acc_code_deps a) (acc_closure_deps b1) b1)
    as (Hc2 & Hn2 & L2 & Hf2 & Hi2 & Hk2).
  { exact Hn1. }
  { intros d Hd. destruct (Hcl1 d Hd) as [Hd' | H]; [ | exact H ].
    apply ok_good. exact (ki_closure_deps a Hinv d Hd'). }
  { intros id cd E. eapply ok_node_mono; [ exact Hn1 | ].
    apply (Hcd id cd); [ exact E | left; reflexivity ]. }
  set (b2 := List.fold_left _ (acc_closure_deps b1) b1) in *.
  clearbody b2.
  set (L3 := map _ _).
  assert (Hi3 : ∀ fct x, fct ∈ L3 -> x ∈ into_nodes fct -> ok_node a b2 x).
  { intros fct x H Hx. unfold L3 in H. into_split H; into_close H Hx.
    ok_tac. }
  cbn [add_facts acc_code_deps acc_facts].
  split; [ congruence | split ].
  - intros id cd fct Hin Hfct Hx. rewrite Hf2, Hf1 in Hfct.
    assert (Hk : kwit a (kw cd))
      by (exists id, cd; split; [ exact Hin | reflexivity ]).
    apply in_app_or in Hfct as [Hfct | Hfct];
      [ exfalso; exact (ok_not_kwit a _ _ Hinv (Hi3 _ _ Hfct Hx) Hk) | ].
    apply in_app_or in Hfct as [Hfct | Hfct];
      [ exfalso; exact (ok_not_kwit a _ _ Hinv (Hi2 _ _ Hfct Hx) Hk) | ].
    apply in_app_or in Hfct as [Hfct | Hfct];
      [ exfalso; exact (ok_not_kwit a _ _ Hinv (Hi1 _ _ Hfct Hx) Hk) | ].
    exact (ki_known a Hinv _ _ _ Hin Hfct Hx).
  - intros b from Hb. rewrite Hf2, Hf1 in Hb.
    apply in_app_or in Hb as [Hb | Hb].
    + exfalso. unfold L3 in Hb. into_split Hb. discriminate Hb.
    + apply in_app_or in Hb as [Hb | Hb].
      * destruct (Hk2 _ _ Hb) as [(id & cd & E & ->) | H].
        -- left. exists id, cd.
           split; [ exact (FlambdaMapProofs.lookup_in _ _ _ E) | reflexivity ].
        -- right. apply in_or_app. right. rewrite Hf2. apply in_or_app.
           left. exact H.
      * exfalso. apply in_app_or in Hb as [Hb | Hb];
          [ exact (Hw1 _ _ Hb) | exact (ki_constructor a Hinv _ _ Hb) ].
Qed.

Lemma kinv_empty : kinv empty_acc.
Proof. constructor; cbn; intros; repeat intro; contradiction. Qed.

(* The facts of the unit put into the witness of known arity of a code dep
   only what its code puts, and their constructors of the field of known
   arity hold the witness of a code dep or a node with any source. *)
Lemma unit_known (flag : reaper_preserve_direct_calls) (u : unit_interface)
    (e : expr) :
  (∀ id cd fct,
     (id, cd) ∈ acc_code_deps (traverse_unit_acc flag u e) ->
     fct ∈ acc_facts (traverse_unit_acc flag u e) ->
     kw cd ∈ into_nodes fct -> known_fact id cd fct)
  ∧ (∀ b from,
       Fact_constructor b known from
       ∈ acc_facts (traverse_unit_acc flag u e) ->
       (∃ id cd,
          (id, cd) ∈ acc_code_deps (traverse_unit_acc flag u e)
          ∧ from = kw cd)
       ∨ Fact_any_source from ∈ acc_facts (traverse_unit_acc flag u e)).
Proof.
  unfold traverse_unit_acc. cbv zeta.
  set (a1 := add_fact (Fact_any_source Node_all_constants) empty_acc).
  assert (Hinv1 : kinv a1).
  { apply (ext_close empty_acc); [ exact kinv_empty | ].
    apply ext_add_fact; [ intros y [<- | []]; ok_tac | ].
    intros ? ? ?; discriminate. }
  destruct (ext_close a1 (snd (fresh a1)) Hinv1 (ext_fresh a1 a1))
    as [Hinv2 Hg2].
  pose proof (ok_close a1 _ _ Hinv1 (ext_fresh a1 a1)
                (ok_fresh a1 a1 (Pos.le_refl _))) as Hr.
  destruct (fresh a1) as [r a2] eqn:E2. cbn [fst snd] in *.
  destruct (ext_close a2 (snd (fresh a2)) Hinv2 (ext_fresh a2 a2))
    as [Hinv3 Hg3].
  pose proof (ok_close a2 _ _ Hinv2 (ext_fresh a2 a2)
                (ok_fresh a2 a2 (Pos.le_refl _))) as Hex.
  destruct (fresh a2) as [ex a3] eqn:E3. cbn [fst snd] in *.
  cbv beta iota zeta.
  set (a4 := add_fact (Fact_any_usage ex) a3).
  assert (H4 : kinv a4 ∧ grows a3 a4).
  { apply (ext_close a3); [ exact Hinv3 | ].
    apply ext_add_fact; [ intros y [] | intros ? ? ?; discriminate ]. }
  destruct H4 as [Hinv4 Hg4].
  match goal with
  | |- context [traverse ?env e a4] => set (env0 := env)
  end.
  assert (Henv : env_good env0 a4).
  { intros k ps x Hk Hx. cbn [env0 env_conts] in Hk.
    destruct Hk as [Hk | [Hk | []]]; injection Hk as <- <-;
      destruct Hx as [<- | []].
    - exact (good_grows _ _ _ Hg4 Hex).
    - exact (good_grows _ _ _ Hg4 (good_grows _ _ _ Hg3 Hr)). }
  destruct (proj1 tstep_traverse e env0 a4 Hinv4 Henv) as [Hinv5 _].
  set (a5 := traverse env0 e a4) in *.
  assert (Hinv6 : kinv (add_fact (Fact_any_usage r) a5)).
  { apply (ext_close a5); [ exact Hinv5 | ].
    apply ext_add_fact; [ intros y [] | intros ? ? ?; discriminate ]. }
  destruct (resolve_known _ (static_names symbols_of_binding e) Hinv6)
    as (Hc & Hk & Hcons).
  rewrite Hc. split; [ exact Hk | exact Hcons ].
Qed.

End KnownWitnesses.

(* ======================================================================== *)
(* [check_known_witnesses]                                                  *)
(* ======================================================================== *)

Abbreviation known := (Field_call_witness Known_arity_code_pointer).

Ltac unfold_rels :=
  cbn [graph_of_facts solution_of_tables holds g_alias g_use g_accessor
       g_constructor g_argument g_parameter g_propagate
       g_alias_if_any_source g_any_usage g_any_source g_any_source_if_used
       g_zero_alloc_source s_alias s_any_usage s_any_source s_has_usage
       s_has_source s_usages s_sources s_reading_field s_escaping_field
       s_field_used] in *.

(* A tuple of the facts of a list is a fact of the list, and back. *)
Ltac tuple_in H :=
  let fct := fresh "fct" in let Hf := fresh "Hf" in let Hy := fresh "Hy" in
  destruct (FlambdaReaperLeastProofs.AutoWidgets.facts_of_list_in _ _ _ H)
    as [fct [Hf Hy]];
  destruct fct; cbn in Hy; try contradiction;
  destruct Hy as [Hy | []]; try discriminate Hy.

Lemma alias_in (l : list fact) (to_ from : node) :
  (to_, from) ∈ f_alias (facts_of_list l) -> Fact_alias to_ from ∈ l.
Proof. intros H. tuple_in H. injection Hy as <- <-. exact Hf. Qed.

Lemma propagate_in (l : list fact) (c to_ from : node) :
  (c, to_, from) ∈ f_propagate (facts_of_list l) ->
  Fact_propagate c to_ from ∈ l.
Proof. intros H. tuple_in H. injection Hy as <- <- <-. exact Hf. Qed.

Lemma if_any_source_in (l : list fact) (c to_ from : node) :
  (c, to_, from) ∈ f_alias_if_any_source (facts_of_list l) ->
  Fact_alias_if_any_source c to_ from ∈ l.
Proof. intros H. tuple_in H. injection Hy as <- <- <-. exact Hf. Qed.

Lemma accessor_in (l : list fact) (to_ base : node) (f : field) :
  (to_, f, base) ∈ f_accessor (facts_of_list l) ->
  Fact_accessor to_ f base ∈ l.
Proof. intros H. tuple_in H. injection Hy as <- <- <-. exact Hf. Qed.

Lemma constructor_in (l : list fact) (base from : node) (f : field) :
  (base, f, from) ∈ f_constructor (facts_of_list l) ->
  Fact_constructor base f from ∈ l.
Proof. intros H. tuple_in H. injection Hy as <- <- <-. exact Hf. Qed.

Lemma parameter_in (l : list fact) (base to_ : node) (i : nat) :
  (base, i, to_) ∈ f_parameter (facts_of_list l) ->
  Fact_parameter base i to_ ∈ l.
Proof. intros H. tuple_in H. injection Hy as <- <- <-. exact Hf. Qed.

Lemma any_source_of (l : list fact) (x : node) :
  Fact_any_source x ∈ l -> x ∈ f_any_source (facts_of_list l).
Proof.
  intros H. cbn [f_any_source facts_of_list]. apply in_flat_map.
  exists (Fact_any_source x). split; [ exact H | left; reflexivity ].
Qed.

Section Known.

Variable a : acc.
Variable is_local : field -> bool.
Variable T : tables.
Hypothesis Hloc : is_local_slots_only is_local.
Hypothesis Hleast : least_solution is_local (facts_of_list (acc_facts a)) T.
Hypothesis Hkf : ∀ id cd fct,
  (id, cd) ∈ acc_code_deps a -> fct ∈ acc_facts a ->
  cd_known_arity_call_witness cd
  ∈ FlambdaReaperLeastProofs.AutoWidgets.into_nodes fct ->
  KnownWitnesses.known_fact id cd fct.
Hypothesis Hcons : ∀ b from,
  Fact_constructor b known from ∈ acc_facts a ->
  (∃ id cd, (id, cd) ∈ acc_code_deps a
            ∧ from = cd_known_arity_call_witness cd)
  ∨ Fact_any_source from ∈ acc_facts a.

Abbreviation F := (facts_of_list (acc_facts a)).
Abbreviation G := (graph_of_facts F).
Abbreviation S := (solution_of_tables T).
Abbreviation D := (derivable is_local G S).

Lemma known_not_local : is_local known = false.
Proof.
  destruct (is_local known) eqn:E; [ | reflexivity ].
  destruct (Hloc _ E) as [(w & Hw) | (fs & Hfs)]; discriminate.
Qed.

(* Nothing is aliased into the witness of known arity of a code dep. *)
Lemma no_alias_into_known (id : code_id) (cd : code_dep) (from : node) :
  (id, cd) ∈ acc_code_deps a ->
  ~ D (Atom_alias (cd_known_arity_call_witness cd) from).
Proof.
  intros Hin Dal. pose proof (fun fct => Hkf id cd fct Hin) as Hk.
  inversion Dal; subst;
    repeat match goal with
    | H : derivable _ _ _ (Atom_reading_field _ _) |- _ =>
      inversion H; subst; clear H
    end;
    unfold_rels;
    match goal with
    | H : (_, _) ∈ f_alias _ |- _ =>
      exact (Hk _ (alias_in _ _ _ H) (or_introl eq_refl))
    | H : (_, _, _) ∈ f_propagate _ |- _ =>
      exact (Hk _ (propagate_in _ _ _ _ H) (or_introl eq_refl))
    | H : (_, _, _) ∈ f_alias_if_any_source _ |- _ =>
      exact (Hk _ (if_any_source_in _ _ _ _ H) (or_introl eq_refl))
    | H : (_, _, _) ∈ f_parameter _ |- _ =>
      exact (Hk _ (parameter_in _ _ _ _ H) (or_intror (or_introl eq_refl))
               eq_refl)
    | H : (_, _, _) ∈ f_accessor _ |- _ =>
      exact (Hk _ (accessor_in _ _ _ _ H) (or_introl eq_refl))
    end.
Qed.

(* The sources of a node only read off [n] with the field of known arity
   are witnesses of known arity of code deps. *)
Lemma known_source (x n s : node) :
  (x, known, n) ∈ f_accessor F ->
  FlambdaReaperLeastProofs.read_only F x known n true ->
  (x, s) ∈ t_sources T ->
  ∃ id cd, (id, cd) ∈ acc_code_deps a ∧ s = cd_known_arity_call_witness cd.
Proof.
  intros Hacc Hro Hs.
  pose proof (FlambdaReaperLeastProofs.least_closed is_local F T Hleast)
    as Hc.
  pose proof known_not_local as Hknown.
  apply (proj2 (mem_pair_In _ _ _)) in Hs.
  apply (FlambdaReaperLeastProofs.least_derivable is_local F T Hleast
           (Atom_sources x s)) in Hs.
  destruct (FlambdaReaperLeastProofs.sources_into is_local F T x known n true
              Hro s Hs) as (from & Dfs & Hgf & Dal & Hgx).
  assert (Hfrom : ∃ b, (b, known, from) ∈ f_constructor F).
  { destruct (FlambdaReaperLeastProofs.alias_into is_local F T x known n true
                Hro Hknown from Dal)
      as [(b & Hb & _) | [(bs & _ & _ & Hb) | [_ Dn]]].
    - exists b. exact Hb.
    - exists bs. exact Hb.
    - exfalso.
      apply (FlambdaReaperLeastProofs.least_holds is_local F T Hleast) in Dn.
      pose proof (any_source_accessor _ _ _ Hc n _ x Dn Hacc Hknown).
      unfold_rels. congruence. }
  destruct Hfrom as (b & Hb).
  destruct (Hcons _ _ (constructor_in _ _ _ _ Hb))
    as [(id & cd & Hin & ->) | Hany].
  - exists id, cd. split; [ exact Hin | ].
    inversion Dfs; subst; [ reflexivity | reflexivity | ].
    exfalso. eapply no_alias_into_known; eassumption.
  - exfalso.
    pose proof (any_source_input _ _ _ Hc from (any_source_of _ _ Hany)) as H.
    unfold_rels. congruence.
Qed.

Lemma known_witnesses :
  check_widgets_read_only F = true ->
  check_known_witnesses F T (acc_code_deps a) = true.
Proof.
  intros Hro. unfold check_widgets_read_only in Hro.
  rewrite forallb_forall in Hro.
  unfold check_known_witnesses. apply andb_true_intro. split.
  - apply forallb_forall. intros [[x f] n] Hacc. cbv beta iota.
    destruct (field_eqb f known) eqn:Ef; [ | reflexivity ].
    apply field_eqb_eq in Ef. subst f. cbn [negb orb].
    pose proof (Hro _ Hacc) as Hrx. cbv beta iota in Hrx.
    rewrite field_eqb_refl in Hrx. cbn [negb orb] in Hrx.
    apply FlambdaReaperLeastProofs.only_read_into_read_only in Hrx.
    apply forallb_forall. intros [x' s] Hs. cbv beta iota.
    destruct (node_eqb x' x) eqn:Ex; [ | reflexivity ].
    apply node_eqb_eq in Ex. subst x'.
    destruct (mem_node s (t_has_source T)); [ | reflexivity ].
    cbn [negb orb].
    destruct (known_source x n s Hacc Hrx Hs) as (id & cd & Hin & ->).
    apply existsb_exists. exists (id, cd). split; [ exact Hin | ].
    apply node_eqb_refl.
  - apply forallb_forall. intros [id cd] Hin. apply forallb_forall.
    intros [[w f] r] Hcw. cbv beta iota.
    destruct (node_eqb w (cd_known_arity_call_witness cd)) eqn:Ew;
      [ | reflexivity ].
    apply node_eqb_eq in Ew. subst w. cbn [negb orb].
    pose proof (Hkf id cd _ Hin (constructor_in _ _ _ _ Hcw)
                  (or_introl eq_refl)) as Hk.
    destruct f as [ | | | | | | | [ i | ] | ]; try reflexivity;
      cbn [KnownWitnesses.known_fact] in Hk.
    + rewrite Hk. apply node_eqb_refl.
    + rewrite Hk. apply node_eqb_refl.
Qed.

End Known.

Theorem traverse_known_witnesses : traverse_known_witnesses_statement.
Proof.
  intros is_local u e T Hloc Hleast.
  destruct (KnownWitnesses.unit_known Auto u e) as [Hkf Hcons].
  pose proof (FlambdaReaperLeastProofs.traverse_widgets_read_only u e)
    as Hro.
  unfold unit_code_deps.
  rewrite traverse_unit_acc_facts in Hleast, Hro |- *.
  exact (known_witnesses _ is_local T Hloc Hleast Hkf Hcons Hro).
Qed.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* The table of [dominated_by_allocation_point] derived from tables passes
   its check. *)
Lemma derive_dominated_checked (T : tables) :
  check_dominated T (derive_dominated T) = true.
Proof.
  unfold check_dominated. apply andb_true_intro. split.
  - apply forallb_forall. intros [y x] Hin.
    exact (proj2 (mem_pair_In _ _ _) Hin).
  - apply forallb_forall. intros [y x] Hin. unfold derive_dominated in Hin.
    apply in_flat_map in Hin. destruct Hin as [[x' y'] [Hs Hin]].
    cbv beta iota in Hin.
    match type of Hin with
    | context [if ?c then _ else _] => destruct c eqn:E; [ | destruct Hin ]
    end.
    destruct Hin as [Heq | []]. injection Heq as <- <-.
    apply andb_true_iff in E as [E1 E2].
    rewrite (proj2 (mem_pair_In _ _ _) Hs), E1, E2. reflexivity.
Qed.

Theorem arm_needs_no_source_auto_end_to_end :
  arm_needs_no_source_auto_end_to_end_statement.
Proof.
  intros is_local u e T max_unbox_size lambda_lifting to_unbox F code_deps
    cannot_change unboxed Hloc Hslots Hfrag Hleast Hunbox Hcount Hkinds.
  exact (FlambdaReaperWrapperShapeProofs.arm_needs_no_source_auto_traversal
           is_local u e T cannot_change max_unbox_size lambda_lifting
           (derive_dominated T) to_unbox Hloc Hslots Hfrag Hleast
           (proj1 (FlambdaReaperCallingConventionsProofs
                     .derive_cannot_change_least F T))
           (derive_dominated_checked T) Hunbox
           (traverse_known_witnesses is_local u e T Hloc Hleast) Hcount
           Hkinds).
Qed.

Theorem kept_result_never_unboxed_auto_end_to_end :
  kept_result_never_unboxed_auto_end_to_end_statement.
Proof.
  intros is_local u e T max_unbox_size lambda_lifting to_unbox F code_deps
    cannot_change unboxed Hloc Hslots Hfrag Hleast Hunbox Hcount Hkinds.
  exact (FlambdaReaperWrapperShapeProofs.kept_result_never_unboxed_traversal
           Auto is_local u e T cannot_change max_unbox_size lambda_lifting
           (derive_dominated T) to_unbox Hloc Hslots Hfrag Hleast
           (proj1 (FlambdaReaperCallingConventionsProofs
                     .derive_cannot_change_least F T))
           (derive_dominated_checked T) Hunbox
           (FlambdaReaperLeastProofs.least_apply_widgets is_local F T Hloc
              Hleast (FlambdaReaperLeastProofs.traverse_widgets_read_only u e))
           (traverse_known_witnesses is_local u e T Hloc Hleast) Hcount
           Hkinds).
Qed.

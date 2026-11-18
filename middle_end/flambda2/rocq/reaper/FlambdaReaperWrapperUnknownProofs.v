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


(* The proofs of FlambdaReaperWrapperUnknownSpec.v. Nothing here needs
   review. *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaWellFormedSpec FlambdaReaperGraphSpec
  FlambdaReaperTraversal FlambdaReaperSolution FlambdaReaperSolutionProofs
  FlambdaReaperRebuild FlambdaReaperCallingConventions FlambdaReaperUnboxing
  FlambdaReaperLeast FlambdaReaperFlowSpec FlambdaReaperLeastSpec
  FlambdaReaperWrapperSpec FlambdaReaperWrapperShapeSpec
  FlambdaReaperWrapperAutoSpec FlambdaReaperWrapperUnknownSpec.
From OxCamlFlambda2 Require FlambdaMapProofs FlambdaReaperFlowProofs
  FlambdaReaperLeastProofs FlambdaReaperWrapperProofs
  FlambdaReaperWrapperShapeProofs FlambdaReaperCallingConventionsProofs
  FlambdaReaperWrapperAutoProofs.
Import ListNotations.

(* ======================================================================== *)
(* The invariant of the traversal                                           *)
(* ======================================================================== *)

(* The nodes read with the field of unknown arity (the widgets) are made
   by calls of unknown arity, and nothing but that read puts anything
   into one; the witnesses of unknown arity of a code dep are made by
   [prepare_code], with a code id, and nothing but constructors of them
   and parameters of which they are the base puts anything into one. The
   invariant of the accumulator ([uinv]), as [uinv] of
   FlambdaReaperWrapperAutoProofs.KnownWitnesses with both sets of nodes
   ([prot]): the facts that put something into a widget or a witness are
   those above; the constructors of the field of unknown arity hold
   witnesses; the nodes that facts put something into are made before
   the counter; the nodes of the deps are not widgets nor witnesses
   ([good]). A step adds facts of neither field of unknown arity
   ([plain]) that put something only into nodes that are not made, good
   at the start of the step or made by it ([ext]); [prepare_code] makes
   the witnesses ([tstep_prepare_code]), a call of unknown arity the
   widgets ([tstep_call_kind]), one group at a time
   ([uinv_apply_deps]). *)
Module UnknownArity.

Import FlambdaReaperLeastProofs.AutoWidgets.

Abbreviation unknown := (Field_call_witness Unknown_arity_code_pointer).
Abbreviation env_conts := FlambdaReaperTraversal.env_conts.
Abbreviation uw := cd_unknown_arity_call_witnesses.

(* [x] is read with the field of unknown arity. *)
Definition uwid (a : acc) (x : node) : Prop :=
  ∃ n, Fact_accessor x unknown n ∈ acc_facts a.

(* [x] is a witness of unknown arity of a code dep. *)
Definition uwit (a : acc) (x : node) : Prop :=
  ∃ id cd, (id, cd) ∈ acc_code_deps a ∧ x ∈ uw cd.

Definition prot (a : acc) (x : node) : Prop := uwid a x ∨ uwit a x.

(* What may put something into a widget read off [n]. *)
Definition wid_fact (fct : fact) (n : node) : Prop :=
  match fct with
  | Fact_accessor _ f base => f = unknown ∧ base = n
  | _ => False
  end.

(* What may put something into a witness [x]. *)
Definition wit_fact (x : node) (fct : fact) : Prop :=
  match fct with
  | Fact_constructor _ _ _ => True
  | Fact_parameter _ _ p => p <> x
  | _ => False
  end.

(* A fact of neither field of unknown arity. *)
Definition plain (fct : fact) : Prop :=
  match fct with
  | Fact_accessor _ f _ | Fact_constructor _ f _ => f <> unknown
  | _ => True
  end.

(* No accessor of the field of unknown arity. *)
Definition no_read (fct : fact) : Prop :=
  match fct with
  | Fact_accessor _ f _ => f <> unknown
  | _ => True
  end.

(* [x] is made before [a] and is neither a widget nor a witness of it. *)
Definition good (a : acc) (x : node) : Prop :=
  made_lt x (acc_next a) ∧ ~ prot a x.

(* [x] is not made, or good in [a0], or made from [a0] to [a]. *)
Definition ok_node (a0 a : acc) (x : node) : Prop :=
  not_made x ∨ good a0 x
  ∨ ∃ k, x = Node_made k ∧ (acc_next a0 <= k)%positive
         ∧ (k < acc_next a)%positive.

Record uinv (a : acc) : Prop := {
  ui_into : ∀ fct x,
    fct ∈ acc_facts a -> x ∈ into_nodes fct -> made_lt x (acc_next a);
  ui_wid_made : ∀ x, uwid a x -> ∃ k, x = Node_made k;
  ui_wid : ∀ x n fct,
    Fact_accessor x unknown n ∈ acc_facts a -> fct ∈ acc_facts a ->
    x ∈ into_nodes fct -> wid_fact fct n;
  ui_wit_made : ∀ id cd x,
    (id, cd) ∈ acc_code_deps a -> x ∈ uw cd ->
    ∃ k, x = Node_made k ∧ (k < acc_next a)%positive;
  ui_wit : ∀ id cd x fct,
    (id, cd) ∈ acc_code_deps a -> x ∈ uw cd -> fct ∈ acc_facts a ->
    x ∈ into_nodes fct -> wit_fact x fct;
  ui_code_id : ∀ id cd x,
    (id, cd) ∈ acc_code_deps a -> x ∈ uw cd ->
    Fact_constructor x Field_code_id_of_call_witness (Node_code_id id)
    ∈ acc_facts a;
  ui_constructor : ∀ b from,
    Fact_constructor b unknown from ∈ acc_facts a -> uwit a from;
  ui_code_deps : ∀ id cd x,
    (id, cd) ∈ acc_code_deps a -> x ∈ code_dep_nodes cd -> good a x;
  ui_apply_deps : ∀ d,
    d ∈ acc_apply_deps a -> good a (ad_apply_call_witness d);
  ui_closure_deps : ∀ d,
    d ∈ acc_closure_deps a -> good a (cld_let_bound_name_of_the_closure d)
}.

Definition env_good (env : traverse_env) (a : acc) : Prop :=
  ∀ k params x, (k, params) ∈ env_conts env -> x ∈ params -> good a x.

(* The widgets and witnesses of [a'] are those of [a] and nodes made from
   [a]. *)
Definition grows (a a' : acc) : Prop :=
  (acc_next a <= acc_next a')%positive
  ∧ ∀ w, prot a' w ->
      prot a w ∨ ∃ k, w = Node_made k ∧ (acc_next a <= k)%positive.

(* From [a] to [a'], plain facts are added that put something into nodes
   that are not made, good in [a0] or made from [a0]; and deps of such
   nodes, but no code dep. *)
Record ext (a0 a a' : acc) : Prop := {
  ext_next : (acc_next a <= acc_next a')%positive;
  ext_facts : ∃ L, acc_facts a' = L ++ acc_facts a
    ∧ (∀ fct x, fct ∈ L -> x ∈ into_nodes fct -> ok_node a0 a' x)
    ∧ (∀ fct, fct ∈ L -> plain fct);
  ext_code_deps : acc_code_deps a' = acc_code_deps a;
  ext_apply_deps : ∀ d, d ∈ acc_apply_deps a' ->
    d ∈ acc_apply_deps a ∨ ok_node a0 a' (ad_apply_call_witness d);
  ext_closure_deps : ∀ d, d ∈ acc_closure_deps a' ->
    d ∈ acc_closure_deps a
    ∨ ok_node a0 a' (cld_let_bound_name_of_the_closure d)
}.

Ltac plain_tac := cbn [plain no_read]; first [ exact I | discriminate
                                              | assumption ].

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
    + intros fct Hin. apply in_app_or in Hin as [Hin | Hin];
        [ exact (Hw2 _ Hin) | exact (Hw1 _ Hin) ].
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
      [ intros fct x [] | intros fct [] ].
  - intros d Hin. left. rewrite <- Hap. exact Hin.
  - intros d Hin. left. rewrite <- Hcl. exact Hin.
Qed.

Lemma ext_refl (a0 a : acc) : ext a0 a a.
Proof. apply ext_same; (reflexivity || lia). Qed.

Lemma ext_add_facts (a0 a : acc) (L : list fact) :
  (∀ fct x, fct ∈ L -> x ∈ into_nodes fct -> ok_node a0 a x) ->
  (∀ fct, fct ∈ L -> plain fct) ->
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
  plain fct ->
  ext a0 a (add_fact fct a).
Proof.
  intros Hi Hw. apply ext_add_facts.
  - intros fct' x [<- | []] Hx. exact (Hi x Hx).
  - intros fct' [<- | []]. exact Hw.
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
      [ intros fct x [] | intros fct [] ].
  - intros d' [<- | Hin]; [ right; exact H | left; exact Hin ].
  - intros d' Hin. left. exact Hin.
Qed.

Lemma ext_push_closure_dep (a0 : acc) (d : closure_dep) (a : acc) :
  ok_node a0 a (cld_let_bound_name_of_the_closure d) ->
  ext a0 a (push_closure_dep d a).
Proof.
  intros H. constructor; cbn; [ lia | | reflexivity | | ].
  - exists []. split; [ reflexivity | split ];
      [ intros fct x [] | intros fct [] ].
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
Lemma prot_made (a : acc) (x : node) :
  uinv a -> prot a x -> ∃ k, x = Node_made k ∧ (k < acc_next a)%positive.
Proof.
  intros Hinv [Hw | (id & cd & Hin & Hx)];
    [ | exact (ui_wit_made a Hinv _ _ _ Hin Hx) ].
  destruct (ui_wid_made a Hinv x Hw) as (k & ->). exists k.
  split; [ reflexivity | ]. destruct Hw as [n Hn].
  exact (ui_into a Hinv _ _ Hn (or_introl eq_refl) k eq_refl).
Qed.

(* The widgets and witnesses of [a'] are those of [a]. *)
Lemma ext_prot (a0 a a' : acc) (x : node) :
  ext a0 a a' -> prot a' x -> prot a x.
Proof.
  intros [_ (L & Hf & _ & Hp) Hc _ _] [[n Hn] | (id & cd & Hin & Hx)].
  - left. exists n. rewrite Hf in Hn. apply in_app_or in Hn as [Hn | Hn];
      [ | exact Hn ].
    specialize (Hp _ Hn). cbn [plain] in Hp. destruct (Hp eq_refl).
  - right. exists id, cd. rewrite <- Hc. auto.
Qed.

Lemma ok_close (a0 a' : acc) (x : node) :
  uinv a0 -> ext a0 a0 a' -> ok_node a0 a' x -> good a' x.
Proof.
  intros Hinv He. pose proof (fun w => ext_prot a0 a0 a' w He) as Hk.
  intros [Hx | [[Hlt Hnw] | (k & -> & H1 & H2)]].
  - split; [ intros k -> ; destruct (Hx k eq_refl) | ].
    intros Hwx. destruct (prot_made a0 x Hinv (Hk x Hwx)) as (k & -> & _).
    exact (Hx k eq_refl).
  - split; [ intros k Hk'; specialize (Hlt k Hk');
             pose proof (ext_next _ _ _ He); lia | ].
    intros Hwx. exact (Hnw (Hk x Hwx)).
  - split; [ intros k' Hk'; injection Hk' as <-; exact H2 | ].
    intros Hwx. destruct (prot_made a0 _ Hinv (Hk _ Hwx)) as (k' & Hk' & Hlt).
    injection Hk' as <-. lia.
Qed.

Lemma ext_close (a0 a' : acc) :
  uinv a0 -> ext a0 a0 a' -> uinv a' ∧ grows a0 a'.
Proof.
  intros Hinv He. pose proof (fun x => ok_close a0 a' x Hinv He) as Hgood.
  pose proof (fun w => ext_prot a0 a0 a' w He) as Hk.
  destruct He as [Hn (L & Hf & Hi & Hp) Hc Hap Hcl].
  assert (Hgood0 : ∀ x, good a0 x -> good a' x).
  { intros x Hx. apply Hgood. right; left. exact Hx. }
  assert (HL : ∀ fct x, fct ∈ L -> x ∈ into_nodes fct -> ~ prot a' x)
    by (intros fct x H Hx; exact (proj2 (Hgood x (Hi _ _ H Hx)))).
  split; [ constructor | split; [ exact Hn | ] ].
  - intros fct x Hin Hx. rewrite Hf in Hin.
    apply in_app_or in Hin as [Hin | Hin].
    + exact (proj1 (Hgood x (Hi _ _ Hin Hx))).
    + intros k Hk'. specialize (ui_into a0 Hinv _ _ Hin Hx k Hk'). lia.
  - intros x [n Hm]. rewrite Hf in Hm. apply in_app_or in Hm as [Hm | Hm];
      [ specialize (Hp _ Hm); cbn [plain] in Hp; destruct (Hp eq_refl) | ].
    exact (ui_wid_made a0 Hinv x (ex_intro _ n Hm)).
  - intros x n fct Hw Hfct Hx.
    assert (Hw' : uwid a' x) by (exists n; exact Hw).
    rewrite Hf in Hw, Hfct. apply in_app_or in Hfct as [Hfct | Hfct];
      [ destruct (HL _ _ Hfct Hx (or_introl Hw')) | ].
    apply in_app_or in Hw as [Hw | Hw];
      [ specialize (Hp _ Hw); cbn [plain] in Hp; destruct (Hp eq_refl) | ].
    exact (ui_wid a0 Hinv _ _ _ Hw Hfct Hx).
  - intros id cd x Hin Hx. rewrite Hc in Hin.
    destruct (ui_wit_made a0 Hinv _ _ _ Hin Hx) as (k & Hk' & Hlt).
    exists k. split; [ exact Hk' | lia ].
  - intros id cd x fct Hin Hx Hfct Hxf.
    assert (Hw' : uwit a' x) by (exists id, cd; auto).
    rewrite Hf in Hfct. apply in_app_or in Hfct as [Hfct | Hfct];
      [ destruct (HL _ _ Hfct Hxf (or_intror Hw')) | ].
    rewrite Hc in Hin. exact (ui_wit a0 Hinv _ _ _ _ Hin Hx Hfct Hxf).
  - intros id cd x Hin Hx. rewrite Hf. apply in_or_app. right.
    rewrite Hc in Hin. exact (ui_code_id a0 Hinv _ _ _ Hin Hx).
  - intros b from Hin. rewrite Hf in Hin.
    apply in_app_or in Hin as [Hin | Hin];
      [ specialize (Hp _ Hin); cbn [plain] in Hp; destruct (Hp eq_refl) | ].
    destruct (ui_constructor a0 Hinv _ _ Hin) as (id & cd & Hcd & Hx).
    exists id, cd. rewrite Hc. auto.
  - intros id cd x Hin Hx. rewrite Hc in Hin.
    exact (Hgood0 _ (ui_code_deps a0 Hinv _ _ _ Hin Hx)).
  - intros d Hin. destruct (Hap _ Hin) as [Hin' | H].
    + exact (Hgood0 _ (ui_apply_deps a0 Hinv _ Hin')).
    + exact (Hgood _ H).
  - intros d Hin. destruct (Hcl _ Hin) as [Hin' | H].
    + exact (Hgood0 _ (ui_closure_deps a0 Hinv _ Hin')).
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
  uinv a -> env_good env a -> uinv a' ∧ grows a a'.

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
  (uinv a -> env_good env a -> ext a a a') -> tstep env a a'.
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

Ltac no_plain :=
  let fct := fresh "fct" in let H := fresh "H" in
  intros fct H; into_split H;
  repeat match goal with y : prod _ _ |- _ => destruct y end;
  subst; plain_tac.

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
    try (plain_tac).
  - intros x [<- | []]. ok_tac.
  - intros x [].
Qed.

Lemma ext_cond_alias (a0 : acc) (env : traverse_env) (from to_ : node)
    (a : acc) :
  ok_node a0 a to_ -> ext a0 a (cond_alias env from to_ a).
Proof.
  intros H. unfold cond_alias.
  destruct (env_current_code_id env) as [id | ]; apply ext_add_fact;
    try (plain_tac); intros x [<- | []]; exact H.
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
      * no_plain.
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
  - no_plain.
Qed.

Lemma ext_traverse_prim (a0 : acc) (env : traverse_env) (x : variable)
    (p : prim) (args : list simple) (a : acc) :
  ext a0 a (traverse_prim env x p args a).
Proof.
  unfold traverse_prim.
  assert (Hacc : ∀ f s b,
             f <> unknown ->
             ext a0 b (add_fact (Fact_accessor (Node_variable x) f
                                   (node_of_simple s)) (note_simple s b))).
  { intros f s b Hf. apply (ext_trans _ _ (note_simple s b));
      [ apply ext_note_simple | ].
    apply ext_add_fact.
    - intros y [<- | []]. ok_tac.
    - cbn [plain]. exact Hf. }
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
      + no_plain. }
  destruct p as [ u | b | v | h | r ];
    [ destruct u | destruct b | destruct v | destruct h | destruct r ];
    repeat match goal with
    | |- ext _ _ (match ?t with _ => _ end) => destruct t
    end;
    try exact Hdefault;
    try (apply ext_block_deps; ok_tac);
    try (apply Hacc; discriminate);
    try apply ext_refl.
  - match goal with
    | |- ext _ _ (add_fact _ ?b) => apply (ext_trans _ _ b)
    end; [ apply Hacc; discriminate | ].
    apply ext_add_fact; [ intros y [<- | []]; ok_tac | ].
    plain_tac.
  - apply (ext_trans _ _ (note_simple s a)); [ apply ext_note_simple | ].
    apply ext_add_fact; [ intros y [<- | []]; ok_tac | ].
    plain_tac.
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
  - no_plain.
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
    plain_tac.
  - apply ext_traverse_prim.
  - apply ext_record_set. intros base code [Heq | []].
    injection Heq as <- _. ok_tac.
  - apply ext_traverse_closure_of_set. exact Hle.
  - apply (ext_trans _ _ (note_simple c a)); [ apply ext_note_simple | ].
    apply ext_add_fact; [ intros y [<- | []]; ok_tac | ].
    plain_tac.
  - apply (ext_trans _ _ (note_simple c a)); [ apply ext_note_simple | ].
    apply ext_add_fact; [ intros y [<- | []]; ok_tac | ].
    plain_tac.
  - apply (ext_trans _ _ (note_simple c a)); [ apply ext_note_simple | ].
    apply ext_add_fact; [ intros y [<- | []]; ok_tac | ].
    plain_tac.
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
    + intros fct H. repeat match goal with
      | H : context [match ?t with _ => _ end] |- _ => destruct t
      end; into_split H; subst; plain_tac.
  - apply ext_add_fact; [ intros y [<- | []]; ok_tac
                        | plain_tac ].
  - apply ext_add_fact; [ intros y [<- | []]; ok_tac
                        | plain_tac ].
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
  apply (ext_trans _ _ _ _ He). apply ext_add_facts; [ | no_plain ].
  intros fct x Hin Hx. into_split Hin; into_close Hin Hx;
    try exact Ho.
  eapply ok_node_mono; [ | apply Hp; eapply indexed_in; eassumption ].
  rewrite Hn. lia.
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
    - apply ext_add_facts; [ | no_plain ].
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
  - apply ext_add_facts; [ | no_plain ].
    intros fct x H Hx. into_split H; into_close H Hx; mono.
  - apply ext_push_apply_dep. cbn [ad_apply_call_witness]. mono.
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
  apply ext_add_facts; [ | no_plain ].
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
  plain_tac.
Qed.


(* ------------------------------------------------------------------------ *)
(* ------------------------------------------------------------------------ *)
(* The witnesses of unknown arity                                           *)
(* ------------------------------------------------------------------------ *)

(* From [a] to [a'], facts are added with no read of the field of unknown
   arity, that put something into nodes that are not made, good in [a0]
   or made from [a0], and into a witness of [ws] only what [wit_fact]
   lets in; their constructors of the field of unknown arity hold
   witnesses of [ws]; and they give each witness of [cs] the code id
   [id]. *)
Definition wfacts (a0 : acc) (id : code_id) (ws cs : list node) (a a' : acc)
    : Prop :=
  (acc_next a <= acc_next a')%positive
  ∧ acc_code_deps a' = acc_code_deps a
  ∧ acc_apply_deps a' = acc_apply_deps a
  ∧ acc_closure_deps a' = acc_closure_deps a
  ∧ ∃ L, acc_facts a' = L ++ acc_facts a
    ∧ (∀ fct x, fct ∈ L -> x ∈ into_nodes fct ->
         ok_node a0 a' x ∧ (x ∈ ws -> wit_fact x fct))
    ∧ (∀ fct, fct ∈ L -> no_read fct)
    ∧ (∀ b from, Fact_constructor b unknown from ∈ L -> from ∈ ws)
    ∧ (∀ w, w ∈ cs ->
         Fact_constructor w Field_code_id_of_call_witness (Node_code_id id)
         ∈ L).

Lemma wfacts_trans (a0 : acc) (id : code_id) (ws cs1 cs2 : list node)
    (a a1 a2 : acc) :
  wfacts a0 id ws cs1 a a1 -> wfacts a0 id ws cs2 a1 a2 ->
  wfacts a0 id ws (cs1 ++ cs2) a a2.
Proof.
  intros (Hn1 & Hc1 & Hap1 & Hcl1 & L1 & Hf1 & Hi1 & Hr1 & Hu1 & Hid1)
    (Hn2 & Hc2 & Hap2 & Hcl2 & L2 & Hf2 & Hi2 & Hr2 & Hu2 & Hid2).
  split; [ lia | split; [ congruence | split; [ congruence | split;
    [ congruence | ] ] ] ].
  exists (L2 ++ L1). split; [ rewrite Hf2, Hf1, app_assoc; reflexivity | ].
  split; [ | split; [ | split ] ].
  - intros fct x H Hx. apply in_app_or in H as [H | H];
      [ exact (Hi2 _ _ H Hx) | ].
    destruct (Hi1 _ _ H Hx) as [Ho Hw].
    split; [ exact (ok_node_mono _ _ _ _ Hn2 Ho) | exact Hw ].
  - intros fct H. apply in_app_or in H as [H | H];
      [ exact (Hr2 _ H) | exact (Hr1 _ H) ].
  - intros b from H. apply in_app_or in H as [H | H];
      [ exact (Hu2 _ _ H) | exact (Hu1 _ _ H) ].
  - intros w H. apply in_app_or in H as [H | H]; apply in_or_app;
      [ right; exact (Hid1 _ H) | left; exact (Hid2 _ H) ].
Qed.

Lemma wfacts_incl (a0 : acc) (id : code_id) (ws cs cs' : list node)
    (a a' : acc) :
  (∀ w, w ∈ cs' -> w ∈ cs) ->
  wfacts a0 id ws cs a a' -> wfacts a0 id ws cs' a a'.
Proof.
  intros Hcs (Hn & Hc & Hap & Hcl & L & Hf & Hi & Hr & Hu & Hid).
  do 4 (split; [ assumption | ]). exists L.
  do 4 (split; [ assumption | ]). intros w Hw. exact (Hid w (Hcs w Hw)).
Qed.

Lemma wfacts_same (a0 : acc) (id : code_id) (ws : list node) (a a' : acc) :
  (acc_next a <= acc_next a')%positive ->
  acc_facts a' = acc_facts a -> acc_code_deps a' = acc_code_deps a ->
  acc_apply_deps a' = acc_apply_deps a ->
  acc_closure_deps a' = acc_closure_deps a -> wfacts a0 id ws [] a a'.
Proof.
  intros Hn Hf Hc Hap Hcl. do 4 (split; [ assumption | ]).
  exists []. rewrite Hf. split; [ reflexivity | ].
  split; [ intros fct x [] | split; [ intros fct [] | split ] ];
    [ intros b from [] | intros w [] ].
Qed.

Lemma wfacts_add (a0 : acc) (id : code_id) (ws cs : list node)
    (L : list fact) (a : acc) :
  (∀ fct x, fct ∈ L -> x ∈ into_nodes fct ->
     ok_node a0 a x ∧ (x ∈ ws -> wit_fact x fct)) ->
  (∀ fct, fct ∈ L -> no_read fct) ->
  (∀ b from, Fact_constructor b unknown from ∈ L -> from ∈ ws) ->
  (∀ w, w ∈ cs ->
     Fact_constructor w Field_code_id_of_call_witness (Node_code_id id)
     ∈ L) ->
  wfacts a0 id ws cs a (add_facts L a).
Proof.
  intros Hi Hr Hu Hid. unfold wfacts.
  cbn [add_facts acc_next acc_code_deps acc_apply_deps acc_closure_deps
    acc_facts].
  split; [ lia | split; [ reflexivity | split; [ reflexivity | split;
    [ reflexivity | ] ] ] ].
  exists L. split; [ reflexivity | ]. auto.
Qed.

Ltac no_read_tac :=
  let fct := fresh "fct" in let H := fresh "H" in
  intros fct H; into_split H;
  repeat match goal with y : prod _ _ |- _ => destruct y end;
  subst; plain_tac.

Ltac no_cons_tac :=
  let b := fresh "b" in let from := fresh "from" in let H := fresh "H" in
  intros b from H; into_split H;
  repeat match goal with y : prod _ _ |- _ => destruct y end;
  try discriminate H.

Lemma made_ne (x y : node) (p : positive) :
  made_lt x p -> (∃ k, y = Node_made k ∧ (p <= k)%positive) -> x <> y.
Proof. intros Hx (k & -> & Hk) ->. specialize (Hx k eq_refl). lia. Qed.

Lemma witness_deps_spec (a0 : acc) (id : code_id) (returns : list node)
    (exn : node) (ws : list node) :
  ∀ (pw : list (list node * node)) (a : acc),
  (acc_next a0 <= acc_next a)%positive ->
  (∀ x, x ∈ ws ->
     ∃ k, x = Node_made k ∧ (acc_next a0 <= k)%positive
          ∧ (k < acc_next a)%positive) ->
  (∀ first w, (first, w) ∈ pw ->
     w ∈ ws ∧ ∀ x, x ∈ first -> made_lt x (acc_next a0) ∧ ok_node a0 a x) ->
  wfacts a0 id ws (map snd pw) a
    (unknown_arity_witness_deps id returns exn pw a).
Proof.
  intros pw. induction pw as [ | [first w] rest IH]; intros a Hle Hws Hpw;
    cbn [unknown_arity_witness_deps map snd];
    [ apply wfacts_same; reflexivity || lia | ].
  destruct (Hpw first w (or_introl eq_refl)) as [Hw Hfirst].
  destruct (Hws w Hw) as (kw & Ekw & Hkw1 & Hkw2).
  assert (Hnw : ∀ x, made_lt x (acc_next a0) -> x <> w).
  { intros x Hx. apply (made_ne x w (acc_next a0) Hx). exists kw. auto. }
  assert (Hnws : ∀ x, made_lt x (acc_next a0) -> x ∉ ws).
  { intros x Hx Hin. destruct (Hws x Hin) as (k & -> & Hk & _).
    specialize (Hx k eq_refl). lia. }
  assert (Hokw : ok_node a0 a w) by (subst w; apply ok_made; lia).
  set (b := add_facts _ a).
  assert (H1 : wfacts a0 id ws [w] a b).
  { apply wfacts_add.
    - intros fct x H Hx. into_split H; into_close H Hx.
      + split; [ exact Hokw | intros _ ].
        cbn [wit_fact]. apply Hnw.
        apply Hfirst. eapply indexed_in. eassumption.
      + match goal with
        | Hy : (?i, ?p) ∈ indexed first |- _ =>
          destruct (Hfirst p (indexed_in _ _ _ Hy)) as [Hlt Hok]
        end.
        split; [ exact Hok | intros Hin; destruct (Hnws _ Hlt Hin) ].
      + split; [ exact Hokw | intros _; exact I ].
    - no_read_tac.
    - no_cons_tac.
    - intros w' [<- | []]. apply in_or_app. right. left. reflexivity. }
  assert (Hbn : acc_next b = acc_next a) by reflexivity.
  replace (w :: map snd rest) with ([w] ++ map snd rest) by reflexivity.
  apply (wfacts_trans _ _ _ _ _ _ _ _ H1).
  destruct rest as [ | [g nw] rest'].
  - cbn [map]. apply wfacts_add.
    + intros fct x H Hx. into_split H; into_close H Hx;
        (split; [ exact (ok_node_mono _ _ _ _ (Pos.le_refl _) Hokw)
                | intros _; exact I ]).
    + no_read_tac.
    + no_cons_tac.
    + intros w' [].
  - destruct (Hpw g nw (or_intror (or_introl eq_refl))) as [Hnw' _].
    clearbody b. destruct (fresh b) as [v c] eqn:Ev.
    assert (Hv : v = Node_made (acc_next b)
                 ∧ acc_next c = Pos.succ (acc_next b)
                 ∧ acc_facts c = acc_facts b
                 ∧ acc_code_deps c = acc_code_deps b
                 ∧ acc_apply_deps c = acc_apply_deps b
                 ∧ acc_closure_deps c = acc_closure_deps b).
    { unfold fresh in Ev. injection Ev as <- <-. repeat split. }
    destruct Hv as (-> & Hcn & Hcf & Hcc & Hcap & Hccl). cbv zeta.
    replace ([] ++ map snd ((g, nw) :: rest'))
      with ([] ++ [] ++ map snd ((g, nw) :: rest')) by reflexivity.
    apply (wfacts_trans _ _ _ [] _ _ c);
      [ apply wfacts_same; [ lia | assumption .. ] | ].
    match goal with
    | |- wfacts _ _ _ _ _ (unknown_arity_witness_deps _ _ _ _ ?d) =>
      apply (wfacts_trans _ _ _ [] _ _ d)
    end.
    + apply wfacts_add.
      * intros fct x H Hx. into_split H; into_close H Hx.
        -- split; [ | intros _; exact I ].
           eapply ok_node_mono; [ | exact Hokw ]. rewrite Hcn, Hbn. lia.
        -- split; [ apply ok_made; rewrite ?Hcn, ?Hbn; lia | ].
           intros Hin. destruct (Hws _ Hin) as (k & Hk & _ & Hlt).
           injection Hk as <-. rewrite Hbn in *. lia.
      * no_read_tac.
      * intros b' from H. destruct H as [H | [H | []]];
          [ discriminate H | injection H as _ <-; exact Hnw' ].
      * intros w' [].
    + apply IH; cbn [add_facts acc_next]; [ rewrite Hcn; lia | | ].
      * intros x Hx. destruct (Hws x Hx) as (k & -> & Hk1 & Hk2).
        exists k. rewrite Hcn. split; [ reflexivity | split; lia ].
      * intros first' w' Hin. destruct (Hpw _ _ (or_intror Hin)) as [H2 H3].
        split; [ exact H2 | ]. intros x Hx. destruct (H3 x Hx) as [H4 H5].
        split; [ exact H4 | ]. eapply ok_node_mono; [ | exact H5 ].
        cbn [add_facts acc_next]. rewrite Hcn, Hbn. lia.
Qed.

Lemma fresh_fields (a : acc) :
  fst (fresh a) = Node_made (acc_next a)
  ∧ acc_next (snd (fresh a)) = Pos.succ (acc_next a)
  ∧ acc_facts (snd (fresh a)) = acc_facts a
  ∧ acc_code_deps (snd (fresh a)) = acc_code_deps a
  ∧ acc_apply_deps (snd (fresh a)) = acc_apply_deps a
  ∧ acc_closure_deps (snd (fresh a)) = acc_closure_deps a.
Proof. repeat split. Qed.

Lemma fresh_list_length (n : nat) (a : acc) :
  length (fst (fresh_list n a)) = n.
Proof.
  revert a. induction n as [ | n IH]; intros a; [ reflexivity | ].
  cbn [fresh_list]. destruct (fresh a) as [x b].
  specialize (IH b). destruct (fresh_list n b) as [xs c]. cbn in *.
  rewrite IH. reflexivity.
Qed.

Lemma map_snd_combine {A B : Type} (l1 : list A) (l2 : list B) :
  length l1 = length l2 -> map snd (combine l1 l2) = l2.
Proof.
  revert l2. induction l1 as [ | x l1 IH]; intros [ | y l2] H;
    cbn in *; try discriminate; [ reflexivity | ].
  rewrite IH by lia. reflexivity.
Qed.

Lemma tupled_spec (a0 : acc) (id : code_id) (params returns : list node)
    (exn : node) (a : acc) :
  (acc_next a0 <= acc_next a)%positive ->
  (∀ x, x ∈ params -> made_lt x (acc_next a0) ∧ ok_node a0 a x) ->
  let r := create_unknown_arity_tupled_call_witnesses id params returns exn a
  in
  wfacts a0 id (fst r) (fst r) a (snd r)
  ∧ ∀ w, w ∈ fst r ->
      ∃ k, w = Node_made k ∧ (acc_next a <= k)%positive
           ∧ (k < acc_next (snd r))%positive.
Proof.
  intros Hle Hp. unfold create_unknown_arity_tupled_call_witnesses.
  destruct (fresh_fields a) as (Ew & Nb & Fb & Cb & Ab & Lb).
  destruct (fresh a) as [w b] eqn:E. cbn [fst snd] in *.
  destruct (fresh_fields b) as (Eu & Nc & Fc & Cc & Ac & Lc).
  destruct (fresh b) as [u c] eqn:E2. cbn [fst snd] in *.
  destruct (fresh_fields c) as (Ek & Nd & Fd & Cd & Ad & Ld).
  destruct (fresh c) as [k d] eqn:E3. cbn [fst snd] in *.
  subst w u k. cbv zeta. cbn [fst snd].
  assert (Hpw : ∀ x, x ∈ params -> x <> Node_made (acc_next a)).
  { intros x Hx. apply (made_ne x _ (acc_next a0) (proj1 (Hp x Hx))).
    exists (acc_next a). split; [ reflexivity | exact Hle ]. }
  split.
  - apply (wfacts_incl _ _ _ ([] ++ [Node_made (acc_next a)]));
      [ intros w H; exact H | ].
    apply (wfacts_trans _ _ _ _ _ _ d);
      [ apply wfacts_same; [ lia | congruence .. ] | ].
    apply wfacts_add.
    + intros fct x H Hx. into_split H; into_close H Hx;
        (split; [ try (apply ok_made; lia) | ]);
        try (intros [Hin | []]; cbn [wit_fact]; try exact I).
      * intros Heq. injection Heq. lia.
      * injection Hin. lia.
      * match goal with
        | Hy : (?i, ?p) ∈ indexed params |- _ =>
          pose proof (indexed_in _ _ _ Hy) as Hpx
        end.
        eapply ok_node_mono; [ | exact (proj2 (Hp _ Hpx)) ]. lia.
      * match goal with
        | Hy : (?i, ?p) ∈ indexed params |- _ =>
          pose proof (indexed_in _ _ _ Hy) as Hpx
        end.
        exact (Hpw _ Hpx (eq_sym Hin)).
      * injection Hin. lia.
      * apply ok_simple || (apply ok_not_made; intros ?k; discriminate).
      * discriminate Hin.
    + no_read_tac.
    + no_cons_tac.
    + intros w [<- | []]. apply in_or_app. right. right. left. reflexivity.
  - intros w [<- | []]. exists (acc_next a).
    cbn [add_facts acc_next]. split; [ reflexivity | lia ].
Qed.

Lemma non_tupled_spec (a0 : acc) (id : code_id) (ar : arity)
    (params returns : list node) (exn : node) (a : acc) :
  (acc_next a0 <= acc_next a)%positive ->
  (∀ x, x ∈ params -> made_lt x (acc_next a0) ∧ ok_node a0 a x) ->
  let r :=
    create_unknown_arity_non_tupled_call_witnesses id ar params returns exn a
  in
  wfacts a0 id (fst r) (fst r) a (snd r)
  ∧ ∀ w, w ∈ fst r ->
      ∃ k, w = Node_made k ∧ (acc_next a <= k)%positive
           ∧ (k < acc_next (snd r))%positive.
Proof.
  intros Hle Hp. unfold create_unknown_arity_non_tupled_call_witnesses.
  cbv zeta.
  set (groups := at_least_one_group (group_by_parameter ar params)).
  destruct (fresh_list_spec (length groups) a)
    as (F1 & F2 & F3 & F4 & F5 & F6).
  pose proof (fresh_list_length (length groups) a) as Hlen.
  destruct (fresh_list (length groups) a) as [ws b] eqn:E.
  cbn [fst snd] in *.
  assert (Hw : wfacts a0 id ws (map snd (combine groups ws)) b
                 (unknown_arity_witness_deps id returns exn
                    (combine groups ws) b)).
  { apply witness_deps_spec; [ lia | | ].
    - intros x Hx. destruct (F6 x Hx) as (m & -> & H1 & H2).
      exists m. split; [ reflexivity | split; lia ].
    - intros first w Hin. split; [ exact (in_combine_r _ _ _ _ Hin) | ].
      intros x Hx.
      destruct (Hp x (group_in ar params first x (in_combine_l _ _ _ _ Hin)
                        Hx)) as [H1 H2].
      split; [ exact H1 | eapply ok_node_mono; [ | exact H2 ]; lia ]. }
  rewrite (map_snd_combine groups ws) in Hw by congruence.
  split.
  - apply (wfacts_incl _ _ _ ([] ++ ws)); [ intros w H; exact H | ].
    apply (wfacts_trans _ _ _ _ _ _ b); [ | exact Hw ].
    apply wfacts_same; assumption.
  - intros w Hin. destruct (F6 w Hin) as (m & -> & H1 & H2).
    exists m. split; [ reflexivity | split; [ lia | ] ].
    destruct Hw as (Hn & _). lia.
Qed.

Lemma create_unknown_spec (a0 : acc) (id : code_id) (is_tupled : bool)
    (ar : arity) (params returns : list node) (exn : node) (a : acc) :
  (acc_next a0 <= acc_next a)%positive ->
  (∀ x, x ∈ params -> made_lt x (acc_next a0) ∧ ok_node a0 a x) ->
  let r :=
    create_unknown_arity_call_witnesses id is_tupled ar params returns exn a
  in
  wfacts a0 id (fst r) (fst r) a (snd r)
  ∧ ∀ w, w ∈ fst r ->
      ∃ k, w = Node_made k ∧ (acc_next a <= k)%positive
           ∧ (k < acc_next (snd r))%positive.
Proof.
  intros Hle Hp. unfold create_unknown_arity_call_witnesses.
  destruct is_tupled; [ apply tupled_spec | apply non_tupled_spec ];
    assumption.
Qed.

(* A node that the facts of a step from [a] may put something into is not
   a widget nor a witness of [a]. *)
Lemma ok_not_prot (a a' : acc) (x : node) :
  uinv a -> ok_node a a' x -> ~ prot a x.
Proof.
  intros Hinv Hx Hk. destruct (prot_made a x Hinv Hk) as (k & -> & Hlt).
  destruct Hx as [Hx | [[_ Hnk] | (k' & Hk' & H1 & _)]].
  - exact (Hx k eq_refl).
  - exact (Hnk Hk).
  - injection Hk' as <-. lia.
Qed.

Lemma ok_made_lt (a0 a : acc) (x : node) :
  (acc_next a0 <= acc_next a)%positive -> ok_node a0 a x ->
  made_lt x (acc_next a).
Proof.
  intros Hle [Hx | [[Hlt _] | (k & -> & _ & H2)]] k' Hk'.
  - destruct (Hx k' Hk').
  - specialize (Hlt k' Hk'). lia.
  - injection Hk' as <-. exact H2.
Qed.

(* ------------------------------------------------------------------------ *)
(* The step that makes a code dep                                           *)
(* ------------------------------------------------------------------------ *)

(* [prepare_code] makes the witnesses of unknown arity, then the code dep;
   the facts before are those of a step. *)
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
  destruct (ext_create_known a id params returns exn b4) as [H5 _];
    [ nexts; lia | exact H4o | ].
  destruct (create_known_arity_call_witness id params returns exn b4)
    as [w b5] eqn:E5. cbn [fst snd] in *. cbv beta iota.
  assert (H05 : ext a a b5).
  { apply (ext_trans _ _ _ _ H1). apply (ext_trans _ _ _ _ He).
    apply (ext_trans _ _ _ _ He0). apply (ext_trans _ _ _ _ H4).
    exact H5. }
  destruct (ext_close a b5 Hinv H05) as [Hinv5 Hg5].
  assert (Hok5 : ∀ x, ok_node a b5 x -> good b5 x)
    by (intros x; exact (ok_close a b5 x Hinv H05)).
  assert (Hp5 : ∀ x, x ∈ params -> good b5 x).
  { intros x Hx. apply Hok5. pose proof (H4o x Hx). mono. }
  assert (Hr5 : ∀ x, x ∈ returns -> good b5 x).
  { intros x Hx. apply Hok5. pose proof (H1o x Hx). mono. }
  assert (He5 : good b5 exn) by (apply Hok5; mono).
  assert (Hm5 : good b5 mc) by (apply Hok5; mono).
  destruct (create_unknown_spec b5 id (code_is_tupled c)
              (code_params_arity c) params returns exn b5)
    as [Hw6 Hr6]; [ lia | | ].
  { intros x Hx. split; [ exact (proj1 (Hp5 x Hx)) | ].
    apply ok_good. exact (Hp5 x Hx). }
  destruct (create_unknown_arity_call_witnesses _ _ _ _ _ _ b5)
    as [ws b6] eqn:E6.
  cbn [fst snd] in *. cbv beta iota zeta.
  destruct Hw6 as (Hn6 & Hc6 & Hap6 & Hcl6 & L & Hf6 & Hi6 & Hnr6 & Hu6
                   & Hid6).
  match goal with
  | |- context [add_code_dep id ?d _] => set (cd := d)
  end.
  assert (Hws : uw cd = ws) by reflexivity.
  set (a' := add_code_dep id cd
               (add_fact (Fact_any_source (Node_code_id id)) b6)).
  assert (Hfa : ∀ fct, fct ∈ acc_facts a' ->
             fct = Fact_any_source (Node_code_id id) ∨ fct ∈ L
             ∨ fct ∈ acc_facts b5).
  { intros fct Hin. cbn [a' add_code_dep add_fact add_facts acc_facts app]
      in Hin.
    rewrite Hf6 in Hin. destruct Hin as [H | Hin]; [ left; auto | right ].
    apply in_app_or in Hin. exact Hin. }
  assert (Hfa' : ∀ fct, fct ∈ L \/ fct ∈ acc_facts b5 ->
             fct ∈ acc_facts a').
  { intros fct H. cbn [a' add_code_dep add_fact add_facts acc_facts app].
    rewrite Hf6. right. apply in_or_app. exact H. }
  assert (Hca : acc_code_deps a' = (id, cd) :: acc_code_deps b5)
    by (cbn [a' add_code_dep add_fact add_facts acc_code_deps];
        rewrite Hc6; reflexivity).
  assert (Hna : acc_next a' = acc_next b6) by reflexivity.
  assert (Hwid : ∀ y, uwid a' y -> uwid b5 y).
  { intros y [n Hm]. destruct (Hfa _ Hm) as [H | [H | H]];
      [ discriminate H | | exists n; exact H ].
    specialize (Hnr6 _ H). cbn [no_read] in Hnr6. destruct (Hnr6 eq_refl). }
  assert (Hprot : ∀ y, prot a' y -> prot b5 y ∨ y ∈ ws).
  { intros y [Hy | (id' & cd' & Hin & Hy)];
      [ left; left; exact (Hwid y Hy) | ].
    rewrite Hca in Hin. destruct Hin as [Heq | Hin].
    - injection Heq as <- <-. right. exact Hy.
    - left. right. exists id', cd'. auto. }
  assert (Hgood : ∀ y, good b5 y -> good a' y).
  { intros y [Hlt Hnp]. split.
    - intros k Hk. specialize (Hlt k Hk). rewrite Hna. lia.
    - intros Hy. destruct (Hprot y Hy) as [H | H]; [ exact (Hnp H) | ].
      destruct (Hr6 y H) as (k & -> & Hk1 & _).
      specialize (Hlt k eq_refl). lia. }
  assert (HL : ∀ fct y, fct ∈ L -> y ∈ into_nodes fct -> ~ prot b5 y).
  { intros fct y H Hy. exact (ok_not_prot b5 b6 y Hinv5
                                (proj1 (Hi6 _ _ H Hy))). }
  assert (Hcode : ∀ y, y ∈ into_nodes (Fact_any_source (Node_code_id id)) ->
             ∀ k, y <> Node_made k).
  { intros y [<- | []] k. discriminate. }
  split; [ constructor | ].
  - intros fct x Hin Hx. rewrite Hna.
    destruct (Hfa _ Hin) as [-> | [H | H]].
    + intros k Hk. destruct (Hcode x Hx k Hk).
    + exact (ok_made_lt _ _ _ Hn6 (proj1 (Hi6 _ _ H Hx))).
    + intros k Hk. specialize (ui_into b5 Hinv5 _ _ H Hx k Hk). lia.
  - intros x Hx. exact (ui_wid_made b5 Hinv5 x (Hwid x Hx)).
  - intros x n fct Hw Hfct Hx.
    assert (Hw5 : Fact_accessor x unknown n ∈ acc_facts b5).
    { destruct (Hfa _ Hw) as [H | [H | H]]; [ discriminate H | | exact H ].
      specialize (Hnr6 _ H). cbn [no_read] in Hnr6.
      destruct (Hnr6 eq_refl). }
    destruct (ui_wid_made b5 Hinv5 x (ex_intro _ n Hw5)) as (k & ->).
    destruct (Hfa _ Hfct) as [-> | [H | H]].
    + destruct (Hcode _ Hx k eq_refl).
    + destruct (HL _ _ H Hx (or_introl (ex_intro _ n Hw5))).
    + exact (ui_wid b5 Hinv5 _ _ _ Hw5 H Hx).
  - intros id' cd' x Hin Hx. rewrite Hca in Hin. rewrite Hna.
    destruct Hin as [Heq | Hin].
    + injection Heq as <- <-. rewrite Hws in Hx.
      destruct (Hr6 x Hx) as (k & -> & _ & Hk). exists k. auto.
    + destruct (ui_wit_made b5 Hinv5 _ _ _ Hin Hx) as (k & -> & Hk).
      exists k. split; [ reflexivity | lia ].
  - intros id' cd' x fct Hin Hx Hfct Hxf. rewrite Hca in Hin.
    destruct Hin as [Heq | Hin].
    + injection Heq as <- <-. rewrite Hws in Hx.
      destruct (Hr6 x Hx) as (k & -> & Hk1 & _).
      destruct (Hfa _ Hfct) as [-> | [H | H]].
      * destruct (Hcode _ Hxf k eq_refl).
      * exact (proj2 (Hi6 _ _ H Hxf) Hx).
      * specialize (ui_into b5 Hinv5 _ _ H Hxf k eq_refl). lia.
    + destruct (ui_wit_made b5 Hinv5 _ _ _ Hin Hx) as (k & -> & _).
      destruct (Hfa _ Hfct) as [-> | [H | H]].
      * destruct (Hcode _ Hxf k eq_refl).
      * destruct (HL _ _ H Hxf (or_intror (ex_intro _ id' (ex_intro _ cd'
                                               (conj Hin Hx))))).
      * exact (ui_wit b5 Hinv5 _ _ _ _ Hin Hx H Hxf).
  - intros id' cd' x Hin Hx. rewrite Hca in Hin. apply Hfa'.
    destruct Hin as [Heq | Hin].
    + injection Heq as <- <-. rewrite Hws in Hx. left. exact (Hid6 x Hx).
    + right. exact (ui_code_id b5 Hinv5 _ _ _ Hin Hx).
  - intros b from Hin. destruct (Hfa _ Hin) as [H | [H | H]];
      [ discriminate H | | ].
    + exists id, cd. rewrite Hca. split; [ left; reflexivity | ].
      rewrite Hws. exact (Hu6 _ _ H).
    + destruct (ui_constructor b5 Hinv5 _ _ H) as (id' & cd' & Hc' & Hx').
      exists id', cd'. rewrite Hca. split; [ right; exact Hc' | exact Hx' ].
  - intros id' cd' x Hin Hx. rewrite Hca in Hin. apply Hgood.
    destruct Hin as [Heq | Hin].
    + injection Heq as <- <-. unfold cd, code_dep_nodes in Hx.
      cbn [cd_my_closure cd_exn cd_return] in Hx.
      destruct Hx as [<- | [<- | Hx]];
        [ exact Hm5 | exact He5 | exact (Hr5 _ Hx) ].
    + exact (ui_code_deps b5 Hinv5 _ _ _ Hin Hx).
  - intros d Hin. apply Hgood. cbn [a' add_code_dep add_fact add_facts
      acc_apply_deps] in Hin. rewrite Hap6 in Hin.
    exact (ui_apply_deps b5 Hinv5 _ Hin).
  - intros d Hin. apply Hgood. cbn [a' add_code_dep add_fact add_facts
      acc_closure_deps] in Hin. rewrite Hcl6 in Hin.
    exact (ui_closure_deps b5 Hinv5 _ Hin).
  - pose proof (ext_next _ _ _ H05) as Ha5. split; [ rewrite Hna; lia | ].
    intros x Hx. destruct (Hprot x Hx) as [H | H].
    + exact (proj2 Hg5 x H).
    + right. destruct (Hr6 x H) as (k & -> & Hk & _). exists k.
      split; [ reflexivity | lia ].
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

(* Adding the read of the field of unknown arity into [x], a made node
   into which no fact puts anything, after facts into good nodes. *)
Lemma uinv_add_widget (a : acc) (L : list fact) (x n : node) :
  uinv a -> (∃ k, x = Node_made k ∧ (k < acc_next a)%positive) ->
  (∀ fct, fct ∈ acc_facts a -> x ∉ into_nodes fct) -> ~ prot a x ->
  (∀ id cd y, (id, cd) ∈ acc_code_deps a -> y ∈ code_dep_nodes cd ->
     y <> x) ->
  (∀ d, d ∈ acc_apply_deps a -> ad_apply_call_witness d <> x) ->
  (∀ d, d ∈ acc_closure_deps a -> cld_let_bound_name_of_the_closure d <> x)
  -> (∀ fct y, fct ∈ L -> y ∈ into_nodes fct -> good a y ∧ y <> x) ->
  (∀ fct, fct ∈ L -> plain fct) ->
  uinv (add_facts (L ++ [Fact_accessor x unknown n]) a)
  ∧ ∀ y, prot (add_facts (L ++ [Fact_accessor x unknown n]) a) y ->
      prot a y ∨ y = x.
Proof.
  intros Hinv (k & -> & Hk) Hnot Hnp Hc Hap Hcl HL Hp.
  set (x := Node_made k) in *.
  set (a' := add_facts (L ++ [Fact_accessor x unknown n]) a).
  assert (Hf : ∀ fct, fct ∈ acc_facts a' ->
             fct ∈ L ∨ fct = Fact_accessor x unknown n ∨ fct ∈ acc_facts a).
  { intros fct H. cbn [a' add_facts acc_facts] in H.
    rewrite <- app_assoc in H. apply in_app_or in H as [H | H];
      [ left; exact H | right ].
    destruct H as [H | H]; [ left; auto | right; exact H ]. }
  assert (Hf' : ∀ fct, fct ∈ acc_facts a -> fct ∈ acc_facts a').
  { intros fct H. cbn [a' add_facts acc_facts]. apply in_or_app.
    right. exact H. }
  assert (Hwid : ∀ y, uwid a' y -> uwid a y ∨ y = x).
  { intros y [m Hm]. destruct (Hf _ Hm) as [H | [H | H]].
    - specialize (Hp _ H). cbn [plain] in Hp. destruct (Hp eq_refl).
    - injection H as -> _. right. reflexivity.
    - left. exists m. exact H. }
  assert (Hprot : ∀ y, prot a' y -> prot a y ∨ y = x).
  { intros y [H | H]; [ destruct (Hwid y H) as [H' | H'];
                        [ left; left; exact H' | right; exact H' ] | ].
    left. right. exact H. }
  assert (Hgood : ∀ y, good a y -> y <> x -> good a' y).
  { intros y [Hlt Hn] Hne. split; [ exact Hlt | ].
    intros Hy. destruct (Hprot y Hy); contradiction. }
  split; [ constructor | exact Hprot ].
  - intros fct y H Hy. cbn [a' add_facts acc_next].
    destruct (Hf _ H) as [H' | [-> | H']].
    + exact (proj1 (proj1 (HL _ _ H' Hy))).
    + destruct Hy as [<- | []]. intros k' Hk'. injection Hk' as <-.
      exact Hk.
    + exact (ui_into a Hinv _ _ H' Hy).
  - intros y Hy. destruct (Hwid y Hy) as [H | ->];
      [ exact (ui_wid_made a Hinv y H) | exists k; reflexivity ].
  - intros y m fct Hw Hfct Hy. destruct (Hf _ Hw) as [H | [H | H]].
    + specialize (Hp _ H). cbn [plain] in Hp. destruct (Hp eq_refl).
    + injection H as -> ->.
      destruct (Hf _ Hfct) as [H' | [-> | H']].
      * destruct (proj2 (HL _ _ H' Hy) eq_refl).
      * cbn [wid_fact]. auto.
      * destruct (Hnot _ H' Hy).
    + destruct (Hf _ Hfct) as [H' | [-> | H']].
      * destruct (proj1 (HL _ _ H' Hy)) as [_ Hn'].
        destruct (Hn' (or_introl (ex_intro _ m H))).
      * destruct Hy as [<- | []].
        destruct (Hnp (or_introl (ex_intro _ m H))).
      * exact (ui_wid a Hinv _ _ _ H H' Hy).
  - exact (ui_wit_made a Hinv).
  - intros id cd y fct Hin Hy Hfct Hyf.
    destruct (Hf _ Hfct) as [H' | [-> | H']].
    + destruct (proj1 (HL _ _ H' Hyf)) as [_ Hn'].
      destruct (Hn' (or_intror (ex_intro _ id (ex_intro _ cd (conj Hin Hy))))).
    + destruct Hyf as [<- | []].
      destruct (Hnp (or_intror (ex_intro _ id (ex_intro _ cd (conj Hin Hy))))).
    + exact (ui_wit a Hinv _ _ _ _ Hin Hy H' Hyf).
  - intros id cd y Hin Hy. apply Hf'. exact (ui_code_id a Hinv _ _ _ Hin Hy).
  - intros b from H. destruct (Hf _ H) as [H' | [H' | H']].
    + specialize (Hp _ H'). cbn [plain] in Hp. destruct (Hp eq_refl).
    + discriminate H'.
    + exact (ui_constructor a Hinv _ _ H').
  - intros id cd y Hin Hy. apply Hgood;
      [ exact (ui_code_deps a Hinv _ _ _ Hin Hy) | exact (Hc _ _ _ Hin Hy) ].
  - intros d Hin. apply Hgood;
      [ exact (ui_apply_deps a Hinv _ Hin) | exact (Hap _ Hin) ].
  - intros d Hin. apply Hgood;
      [ exact (ui_closure_deps a Hinv _ Hin) | exact (Hcl _ Hin) ].
Qed.

Lemma fresh_list_nodup (n : nat) (a : acc) : NoDup (fst (fresh_list n a)).
Proof.
  revert a. induction n as [ | n IH]; intros a; [ constructor | ].
  cbn [fresh_list fresh]. cbv beta iota.
  match goal with
  | |- context [fresh_list n ?b] =>
    pose proof (IH b) as H1; destruct (fresh_list_spec n b) as (_ & _ & _
      & _ & _ & H2); destruct (fresh_list n b) as [xs c]
  end.
  cbn [fst snd acc_next] in *. constructor; [ | exact H1 ].
  intros Hin. destruct (H2 _ Hin) as (m & Hm & Hm1 & _).
  injection Hm as <-. lia.
Qed.

Lemma good_made_ne (a0 : acc) (x y : node) :
  good a0 x -> (∃ k, y = Node_made k ∧ (acc_next a0 <= k)%positive) ->
  x <> y.
Proof. intros [Hlt _] H. exact (made_ne x y _ Hlt H). Qed.

(* The reads of a call of unknown arity, one group at a time: each node
   read with the field of unknown arity is a witness of the call after
   the first. *)
Lemma uinv_apply_deps (a0 : acc) (returns : list node) (exn called : node) :
  uinv a0 ->
  ∀ (aw : list (list simple * node)) (a : acc),
  uinv a ->
  acc_code_deps a = acc_code_deps a0 ->
  acc_apply_deps a = acc_apply_deps a0 ->
  acc_closure_deps a = acc_closure_deps a0 ->
  (∀ x, x ∈ exn :: called :: returns -> good a x) ->
  (∀ first w, (first, w) ∈ aw ->
     ∃ k, w = Node_made k ∧ (acc_next a0 <= k)%positive
          ∧ (k < acc_next a)%positive) ->
  (∀ w, w ∈ map snd aw -> w ∉ exn :: called :: returns) ->
  (∀ w, w ∈ tl (map snd aw) ->
     (∀ fct, fct ∈ acc_facts a -> w ∉ into_nodes fct) ∧ ~ prot a w) ->
  NoDup (map snd aw) ->
  uinv (unknown_arity_apply_deps returns exn called aw a)
  ∧ acc_code_deps (unknown_arity_apply_deps returns exn called aw a)
    = acc_code_deps a0
  ∧ acc_apply_deps (unknown_arity_apply_deps returns exn called aw a)
    = acc_apply_deps a0
  ∧ acc_closure_deps (unknown_arity_apply_deps returns exn called aw a)
    = acc_closure_deps a0
  ∧ (acc_next a
     <= acc_next (unknown_arity_apply_deps returns exn called aw a))%positive
  ∧ ∀ y, prot (unknown_arity_apply_deps returns exn called aw a) y ->
      prot a y ∨ y ∈ tl (map snd aw).
Proof.
  intros Hinv0 aw. induction aw as [ | [first w] rest IH];
    intros a Hinv Hc Hap Hcl Hgood Hmade Hdist Htl Hnd;
    cbn [unknown_arity_apply_deps].
  { split; [ exact Hinv | split; [ exact Hc | split; [ exact Hap | ] ] ].
    split; [ exact Hcl | split; [ lia | ] ]. intros y Hy. left. exact Hy. }
  set (b := add_facts _ a).
  assert (Hab : ext a a b).
  { apply ext_add_facts; [ | no_plain ].
    intros fct x H Hx. into_split H; into_close H Hx; apply ok_good;
      apply Hgood; [ left | right; left ]; reflexivity. }
  destruct (ext_close a b Hinv Hab) as [Hinvb Hgb].
  assert (Hpb : ∀ y, prot b y -> prot a y)
    by (intros y; exact (ext_prot a a b y Hab)).
  assert (Hfb : ∀ fct y, fct ∈ acc_facts b -> y ∈ into_nodes fct ->
             y = exn ∨ y = called ∨ fct ∈ acc_facts a).
  { intros fct y H Hy. cbn [b add_facts acc_facts] in H.
    apply in_app_or in H as [H | H]; [ | right; right; exact H ].
    into_split H; into_close H Hy; auto. }
  assert (Hdeps : ∀ y, (∃ k, y = Node_made k ∧ (acc_next a0 <= k)%positive)
             -> (∀ id cd z, (id, cd) ∈ acc_code_deps b ->
                   z ∈ code_dep_nodes cd -> z <> y)
                ∧ (∀ d, d ∈ acc_apply_deps b -> ad_apply_call_witness d <> y)
                ∧ (∀ d, d ∈ acc_closure_deps b ->
                     cld_let_bound_name_of_the_closure d <> y)).
  { intros y Hy. cbn [b add_facts acc_code_deps acc_apply_deps
      acc_closure_deps]. rewrite Hc, Hap, Hcl.
    split; [ | split ].
    - intros id cd z Hin Hz.
      exact (good_made_ne _ _ _ (ui_code_deps a0 Hinv0 _ _ _ Hin Hz) Hy).
    - intros d Hin. exact (good_made_ne _ _ _ (ui_apply_deps a0 Hinv0 _ Hin)
                             Hy).
    - intros d Hin.
      exact (good_made_ne _ _ _ (ui_closure_deps a0 Hinv0 _ Hin) Hy). }
  destruct rest as [ | [g nw] rest'].
  - set (a' := add_facts _ b).
    assert (Hba : ext b b a').
    { apply ext_add_facts; [ | no_plain ].
      intros fct x H Hx. into_split H; into_close H Hx. apply ok_good.
      apply (good_grows _ _ _ Hgb). apply Hgood. right; right.
      eapply indexed_in. eassumption. }
    destruct (ext_close b a' Hinvb Hba) as [Hinv' _].
    split; [ exact Hinv' | ].
    split; [ exact Hc | split; [ exact Hap | split; [ exact Hcl | ] ] ].
    split; [ cbn [a' b add_facts acc_next]; lia | ].
    intros y Hy. left. exact (Hpb y (ext_prot _ _ _ y Hba Hy)).
  - cbn [map snd tl] in Hdist, Htl, Hnd.
    apply NoDup_cons_iff in Hnd as [Hnd1 Hnd].
    apply NoDup_cons_iff in Hnd as [Hnd2 Hnd3].
    destruct (Hmade g nw (or_intror (or_introl eq_refl)))
      as (knw & Enw & Hknw1 & Hknw2).
    destruct (fresh_fields b) as (Ev & Nc & Fc & Cc & Apc & Clc).
    destruct (fresh b) as [v c] eqn:Ec. cbn [fst snd] in *. subst v.
    assert (Hbc : ext a b c)
      by (pose proof (ext_fresh a b) as H; rewrite Ec in H; exact H).
    destruct (ext_close a c Hinv (ext_trans _ _ _ _ Hab Hbc))
      as [Hinvc Hgc].
    assert (Hpc : ∀ y, prot c y -> prot a y)
      by (intros y Hy; exact (ext_prot _ _ _ y (ext_trans _ _ _ _ Hab Hbc)
                                Hy)).
    assert (Hgv : good c (Node_made (acc_next b))).
    { apply (ok_close a c _ Hinv (ext_trans _ _ _ _ Hab Hbc)).
      apply ok_made; [ cbn [b add_facts acc_next]; lia | rewrite Nc; lia ]. }
    assert (Hvnw : Node_made (acc_next b) <> nw).
    { rewrite Enw. intros Heq. injection Heq as Heq.
      cbn [b add_facts acc_next] in Heq. lia. }
    assert (Hnwc : ¬ prot c nw).
    { intros H. exact (proj2 (Htl nw (or_introl eq_refl)) (Hpc _ H)). }
    destruct (Hdeps nw (ex_intro _ knw (conj Enw Hknw1))) as (D1 & D2 & D3).
    cbv zeta.
    destruct (uinv_add_widget c
                [Fact_accessor (Node_made (acc_next b))
                   (Field_return_of_call (Normal 0)) w]
                nw (Node_made (acc_next b)))
      as [Hinvd Hpd].
    { exact Hinvc. }
    { exists knw. split; [ exact Enw | rewrite Nc; cbn [b add_facts acc_next];
                           lia ]. }
    { intros fct Hfct Hx. rewrite Fc in Hfct.
      destruct (Hfb _ _ Hfct Hx) as [Heq | [Heq | H]].
      - apply (Hdist nw (or_intror (or_introl eq_refl))). left.
        exact (eq_sym Heq).
      - apply (Hdist nw (or_intror (or_introl eq_refl))). right; left.
        exact (eq_sym Heq).
      - exact (proj1 (Htl nw (or_introl eq_refl)) _ H Hx). }
    { exact Hnwc. }
    { rewrite Cc. exact D1. }
    { rewrite Apc. exact D2. }
    { rewrite Clc. exact D3. }
    { intros fct y [<- | []] [<- | []]. split; [ exact Hgv | exact Hvnw ]. }
    { intros fct [<- | []]. plain_tac. }
    cbn [app] in Hinvd, Hpd.
    set (d := add_facts _ c) in *.
    assert (Hpd' : ∀ y, prot d y -> prot a y ∨ y = nw).
    { intros y Hy. destruct (Hpd y Hy) as [H | H];
        [ left; exact (Hpc y H) | right; exact H ]. }
    assert (Hnd' : acc_next d = Pos.succ (acc_next b))
      by (cbn [d add_facts acc_next]; exact Nc).
    destruct (IH d) as (Hinv' & Hc' & Hap' & Hcl' & Hn' & Hp').
    + exact Hinvd.
    + cbn [d add_facts acc_code_deps]. rewrite Cc. exact Hc.
    + cbn [d add_facts acc_apply_deps]. rewrite Apc. exact Hap.
    + cbn [d add_facts acc_closure_deps]. rewrite Clc. exact Hcl.
    + intros x Hx. destruct (Hgood x Hx) as [Hlt Hnp]. split.
      * intros k Hk. specialize (Hlt k Hk). rewrite Hnd'.
        cbn [b add_facts acc_next]. lia.
      * intros H. destruct (Hpd' x H) as [H' | ->]; [ exact (Hnp H') | ].
        exact (Hdist nw (or_intror (or_introl eq_refl)) Hx).
    + intros first' w' Hin. destruct (Hmade _ _ (or_intror Hin))
        as (k & -> & Hk1 & Hk2).
      exists k. split; [ reflexivity | split; [ exact Hk1 | ] ].
      rewrite Hnd'. cbn [b add_facts acc_next]. lia.
    + intros w' Hw'. exact (Hdist w' (or_intror Hw')).
    + intros w' Hw'. cbn [map snd tl] in Hw'.
      assert (Hne : w' <> nw) by (intros ->; exact (Hnd2 Hw')).
      assert (Htl' := Htl w' (or_intror Hw')).
      split.
      * intros fct Hfct Hx. cbn [d add_facts acc_facts app] in Hfct.
        destruct Hfct as [<- | [<- | Hfct]].
        -- destruct Hx as [Hx | []].
           apply in_map_iff in Hw' as [[f' w''] [Hw'' Hin']].
           cbn [snd] in Hw''. subst w''.
           destruct (Hmade _ _ (or_intror (or_intror Hin')))
             as (k & -> & _ & Hk2).
           injection Hx as Hx. cbn [b add_facts acc_next] in Hx. lia.
        -- destruct Hx as [<- | []]. exact (Hne eq_refl).
        -- rewrite Fc in Hfct.
           destruct (Hfb _ _ Hfct Hx) as [Heq | [Heq | H]].
           ++ apply (Hdist w' (or_intror (or_intror Hw'))). left.
              exact (eq_sym Heq).
           ++ apply (Hdist w' (or_intror (or_intror Hw'))). right; left.
              exact (eq_sym Heq).
           ++ exact (proj1 Htl' _ H Hx).
      * intros H. destruct (Hpd' w' H) as [H' | H'];
          [ exact (proj2 Htl' H') | exact (Hne H') ].
    + exact (NoDup_cons _ Hnd2 Hnd3).
    + split; [ exact Hinv' | split; [ exact Hc' | split; [ exact Hap' | ] ] ].
      split; [ exact Hcl' | split ].
      * rewrite Hnd' in Hn'. cbn [b add_facts acc_next] in Hn'. lia.
      * intros y Hy. destruct (Hp' y Hy) as [H | H].
        -- destruct (Hpd' y H) as [H' | ->]; [ left; exact H' | ].
           right. left. reflexivity.
        -- right. right. exact H.
Qed.

(* The step that makes widgets: the call of unknown arity. *)
Lemma tstep_call_kind (env : traverse_env) (ap : apply_expr)
    (exn_arg : node) (return_args : list node) (a : acc) :
  uinv a -> (∀ x, x ∈ exn_arg :: return_args -> good a x) ->
  uinv (traverse_call_kind env ap exn_arg return_args a)
  ∧ grows a (traverse_call_kind env ap exn_arg return_args a).
Proof.
  intros Hinv Hgood. unfold traverse_call_kind. cbv zeta.
  pose proof (ext_note_simple a (ap_callee ap) a) as H1.
  destruct (note_simple_fields (ap_callee ap) a) as (N1 & N2 & N3 & N4 & N5).
  set (a1 := note_simple (ap_callee ap) a) in *. clearbody a1.
  set (closure := node_of_simple (ap_callee ap)).
  assert (Hin1 : ∀ x, x ∈ exn_arg :: return_args -> ok_node a a1 x).
  { intros x Hx. apply ok_good. exact (Hgood x Hx). }
  destruct (ap_call_kind ap) as [id | ].
  - apply (ext_close a); [ exact Hinv | ].
    apply (ext_trans _ _ _ _ H1).
    destruct (ext_make_known a env ap return_args exn_arg a1)
      as [Hk Hcw]; [ nexts; lia | exact Hin1 | ].
    destruct (make_known_arity_apply_widget env ap return_args exn_arg a1)
      as [cw a2] eqn:E. cbn [fst snd] in *. cbv beta iota zeta.
    set (a3 := add_fact (Fact_accessor cw _ closure) a2).
    assert (H3 : ext a a2 a3).
    { apply ext_add_fact; [ intros y [<- | []]; exact Hcw | plain_tac ]. }
    apply (ext_trans _ _ _ _ Hk). apply (ext_trans _ _ _ _ H3).
    assert (Hcw3 : ok_node a a3 cw) by mono.
    clearbody a3.
    destruct (env_should_preserve_direct_calls env);
      [ | destruct (negb (among id (env_unit_code_ids env))) | ];
      try apply ext_refl; apply ext_add_apply; try exact Hcw3; nexts; lia.
  - unfold make_unknown_arity_apply_widget.
    destruct (fresh_fields a1) as (Ec & Nb & Fb & Cb & Apb & Clb).
    destruct (fresh a1) as [called b] eqn:Eb. cbn [fst snd] in *.
    subst called. cbv beta iota zeta.
    set (called := Node_made (acc_next a1)).
    set (b1 := add_fact (Fact_any_usage called) b).
    set (groups := at_least_one_group _).
    destruct (fresh_list_spec (length groups) b1)
      as (F1 & F2 & F3 & F4 & F5 & F6).
    pose proof (fresh_list_length (length groups) b1) as Hlen.
    pose proof (fresh_list_nodup (length groups) b1) as Hnd.
    pose proof (ext_fresh_list a (length groups) b1) as Hfl.
    destruct (fresh_list (length groups) b1) as [ws b2] eqn:E2.
    cbn [fst snd] in *. cbv beta iota zeta.
    destruct (note_simples_fields (ap_args ap) b2) as (M1 & M2 & M3 & M4 & M5).
    pose proof (ext_note_simples a (ap_args ap) b2) as H4.
    set (b3 := note_simples (ap_args ap) b2) in *. clearbody b3.
    assert (Hb1 : acc_next b1 = Pos.succ (acc_next a))
      by (cbn [b1 add_fact add_facts acc_next]; rewrite Nb; congruence).
    assert (H03 : ext a a b3).
    { apply (ext_trans _ _ _ _ H1). apply (ext_trans _ _ b).
      { pose proof (ext_fresh a a1) as H. rewrite Eb in H. exact H. }
      apply (ext_trans _ _ b1).
      { apply ext_add_fact; [ intros y [] | plain_tac ]. }
      apply (ext_trans _ _ b2); [ | exact H4 ].
      apply (proj1 (Hfl ltac:(rewrite Hb1; lia))). }
    destruct (ext_close a b3 Hinv H03) as [Hinv3 Hg3].
    assert (Hf3 : acc_facts b3 = Fact_any_usage called :: acc_facts a).
    { rewrite M2, F1. cbn [b1 add_fact add_facts acc_facts app].
      rewrite Fb, N2. reflexivity. }
    assert (Hws : ∀ w, w ∈ ws ->
               ∃ k, w = Node_made k ∧ (Pos.succ (acc_next a) <= k)%positive
                    ∧ (k < acc_next b3)%positive).
    { intros w Hw. destruct (F6 w Hw) as (k & -> & Hk1 & Hk2).
      exists k. rewrite M1. rewrite <- Hb1. auto. }
    assert (Hws' : ∀ w, w ∈ ws -> ∀ x, made_lt x (acc_next a) -> x <> w).
    { intros w Hw x Hx. destruct (Hws w Hw) as (k & -> & Hk1 & _).
      apply (made_ne x _ (acc_next a) Hx). exists k. split; [ reflexivity | ].
      lia. }
    assert (Hcalled : good b3 called).
    { apply (ok_close a b3 _ Hinv H03). apply ok_made; [ rewrite N1; lia | ].
      rewrite M1, N1. rewrite Hb1 in F5. lia. }
    assert (Hmap : map snd (combine groups ws) = ws)
      by (apply map_snd_combine; congruence).
    assert (Hcw : ∀ w, w ∈ ws -> called <> w).
    { intros w Hw Heq. destruct (Hws w Hw) as (k & Hk & Hk1 & _).
      rewrite <- Heq in Hk. injection Hk as Hk. rewrite N1 in Hk. lia. }
    destruct (uinv_apply_deps a return_args exn_arg called Hinv
                (combine groups ws) b3 Hinv3)
      as (Hinv4 & Hc4 & Hap4 & Hcl4 & Hn4 & Hp4).
    { exact (ext_code_deps _ _ _ H03). }
    { rewrite M4, F3. cbn [b1 add_fact add_facts acc_apply_deps].
      rewrite Apb, N4. reflexivity. }
    { rewrite M5, F4. cbn [b1 add_fact add_facts acc_closure_deps].
      rewrite Clb, N5. reflexivity. }
    { intros x [<- | [<- | Hx]].
      - exact (good_grows _ _ _ Hg3 (Hgood _ (or_introl eq_refl))).
      - exact Hcalled.
      - exact (good_grows _ _ _ Hg3 (Hgood _ (or_intror Hx))). }
    { intros first w Hin. destruct (Hws w (in_combine_r _ _ _ _ Hin))
        as (k & -> & Hk1 & Hk2).
      exists k. split; [ reflexivity | split; lia ]. }
    { intros w Hw. rewrite Hmap in Hw. intros [Heq | [Heq | Hx]].
      - exact (Hws' w Hw exn_arg (proj1 (Hgood _ (or_introl eq_refl))) Heq).
      - exact (Hcw w Hw Heq).
      - exact (Hws' w Hw _ (proj1 (Hgood _ (or_intror Hx))) eq_refl). }
    { intros w Hw. rewrite Hmap in Hw.
      assert (Hw' : w ∈ ws)
        by (destruct ws; [ destruct Hw | right; exact Hw ]).
      split.
      - intros fct Hfct Hx. rewrite Hf3 in Hfct.
        destruct Hfct as [<- | Hfct]; [ destruct Hx | ].
        exact (Hws' w Hw' w (ui_into a Hinv _ _ Hfct Hx) eq_refl).
      - intros Hp. destruct (prot_made a w Hinv (ext_prot _ _ _ w H03 Hp))
          as (k & -> & Hk).
        destruct (Hws _ Hw') as (k' & Hk' & H1' & _).
        injection Hk' as <-. lia. }
    { rewrite Hmap. exact Hnd. }
    rewrite Hmap in Hp4.
    set (b4 := unknown_arity_apply_deps _ _ _ _ b3) in *. clearbody b4.
    destruct (fresh_fields b4) as (Ean & N6 & F6' & C6 & Ap6 & Cl6).
    pose proof (ext_fresh b4 b4) as H45.
    destruct (fresh b4) as [an b5] eqn:E5. cbn [fst snd] in *. subst an.
    destruct (ext_close b4 b5 Hinv4 H45) as [Hinv5 Hg5].
    destruct ws as [ | first rest].
    { exfalso. cbn [length] in Hlen. unfold groups, at_least_one_group in Hlen.
      destruct (group_by_parameter _ _); discriminate Hlen. }
    cbv beta iota.
    assert (Hfirst : good b5 first).
    { destruct (Hws first (or_introl eq_refl)) as (k & Hk & Hk1 & Hk2).
      split.
      - rewrite Hk. intros k' Hk'. injection Hk' as <-. rewrite N6. lia.
      - intros H. destruct (Hp4 first (ext_prot _ _ _ _ H45 H)) as [H' | H'].
        + destruct (prot_made a first Hinv (ext_prot _ _ _ _ H03 H'))
            as (k' & Hk' & Hlt).
          rewrite Hk in Hk'. injection Hk' as <-. lia.
        + apply NoDup_cons_iff in Hnd as [Hnd _]. exact (Hnd H'). }
    set (b6 := cond_alias env (Node_made (acc_next b4)) first b5).
    assert (H56 : ext b5 b5 b6) by (apply ext_cond_alias; apply ok_good;
                                    exact Hfirst).
    destruct (ext_close b5 b6 Hinv5 H56) as [Hinv6 Hg6].
    assert (Hb6 : acc_next b6 = acc_next b5
                  ∧ acc_code_deps b6 = acc_code_deps b5
                  ∧ acc_apply_deps b6 = acc_apply_deps b5
                  ∧ acc_closure_deps b6 = acc_closure_deps b5
                  ∧ ∀ fct, fct ∈ acc_facts b6 ->
                      (∀ y, y ∈ into_nodes fct -> y = first)
                      ∨ fct ∈ acc_facts b5).
    { unfold b6, cond_alias.
      destruct (env_current_code_id env); cbn; (split; [ reflexivity | ]);
        (split; [ reflexivity | ]); (split; [ reflexivity | ]);
        (split; [ reflexivity | ]); intros fct [<- | H];
        try (right; exact H); left; intros y [<- | []]; reflexivity. }
    destruct Hb6 as (Hn6 & Hc6 & Hap6 & Hcl6 & Hf6).
    clearbody b6.
    assert (Hlt4 : ∀ x, made_lt x (acc_next b4) -> x <> Node_made (acc_next b4))
      by (intros x Hx ->; specialize (Hx _ eq_refl); lia).
    assert (Hfn : first <> Node_made (acc_next b4)).
    { apply Hlt4. destruct (Hws first (or_introl eq_refl)) as (k & -> & _ & Hk).
      intros k' Hk'. injection Hk' as <-. lia. }
    assert (Hdep : ∀ x, good a x -> x <> Node_made (acc_next b4)).
    { intros x [Hx _]. apply Hlt4. intros k Hk. specialize (Hx k Hk).
      pose proof (ext_next _ _ _ H03). lia. }
    match goal with
    | |- context [add_fact (Fact_accessor ?x unknown ?n) b6] =>
      change (add_fact (Fact_accessor x unknown n) b6)
        with (add_facts ([] ++ [Fact_accessor x unknown n]) b6)
    end.
    destruct (uinv_add_widget b6 [] (Node_made (acc_next b4)) closure Hinv6)
      as [Hinvf Hpf].
    { exists (acc_next b4). split; [ reflexivity | rewrite Hn6, N6; lia ]. }
    { intros fct Hfct Hx. destruct (Hf6 _ Hfct) as [H | H].
      - exact (Hfn (eq_sym (H _ Hx))).
      - rewrite F6' in H. exact (Hlt4 _ (ui_into b4 Hinv4 _ _ H Hx) eq_refl). }
    { intros H. destruct (prot_made b4 _ Hinv4
                            (ext_prot _ _ _ _ H45 (ext_prot _ _ _ _ H56 H)))
        as (k & Hk & Hlt).
      injection Hk as <-. lia. }
    { intros id cd y Hin Hy. rewrite Hc6, C6, Hc4 in Hin.
      exact (Hdep _ (ui_code_deps a Hinv _ _ _ Hin Hy)). }
    { intros d Hin. rewrite Hap6, Ap6, Hap4 in Hin.
      exact (Hdep _ (ui_apply_deps a Hinv _ Hin)). }
    { intros d Hin. rewrite Hcl6, Cl6, Hcl4 in Hin.
      exact (Hdep _ (ui_closure_deps a Hinv _ Hin)). }
    { intros fct y []. }
    { intros fct []. }
    split; [ exact Hinvf | split ].
    + cbn [add_facts acc_next]. rewrite Hn6, N6.
      pose proof (ext_next _ _ _ H03). lia.
    + intros y Hy. destruct (Hpf y Hy) as [H | ->].
      * destruct (Hp4 y (ext_prot _ _ _ _ H45 (ext_prot _ _ _ _ H56 H)))
          as [H' | H'].
        -- exact (proj2 Hg3 y H').
        -- right. destruct (Hws y (or_intror H')) as (k & -> & Hk & _).
           exists k. split; [ reflexivity | lia ].
      * right. exists (acc_next b4). split; [ reflexivity | ].
        pose proof (ext_next _ _ _ H03). lia.
Qed.

Lemma tstep_traverse_apply (env : traverse_env) (ap : apply_expr) (a : acc) :
  tstep env a (traverse_apply env ap a).
Proof.
  intros Hinv Henv. unfold traverse_apply.
  destruct (ext_find_cont a env (ap_continuation ap)
              (length (unarize (ap_return_arity ap))) a Henv)
    as [H1 H1o]; [ lia | ].
  destruct (find_cont env (ap_continuation ap) _ a) as [rets b1] eqn:E1.
  cbn [fst snd] in *. cbv beta iota.
  destruct (ext_find_cont a env (ap_exn_continuation ap)
              (S (length (ap_exn_extra_args ap))) b1 Henv)
    as [H2 H2o]; [ nexts; lia | ].
  destruct (find_cont env (ap_exn_continuation ap) _ b1) as [eps b2] eqn:E2.
  cbn [fst snd] in *. cbv beta iota.
  assert (Hpre : ∀ (exn_arg : node) (b3 : acc),
            ext a b2 b3 -> ok_node a b3 exn_arg ->
            uinv (traverse_call_kind env ap exn_arg rets b3)
            ∧ grows a (traverse_call_kind env ap exn_arg rets b3)).
  { intros exn_arg b3 H3 Hex.
    assert (H03 : ext a a b3)
      by exact (ext_trans _ _ _ _ (ext_trans _ _ _ _ H1 H2) H3).
    destruct (ext_close a b3 Hinv H03) as [Hinv3 Hg3].
    destruct (tstep_call_kind env ap exn_arg rets b3 Hinv3)
      as [Hinv4 Hg4].
    - intros x [<- | Hx]; apply (ok_close a); try assumption.
      pose proof (H1o x Hx). nexts. eapply ok_node_mono; [ | eassumption ].
      lia.
    - split; [ exact Hinv4 | exact (grows_trans _ _ _ Hg3 Hg4) ]. }
  destruct eps as [ | ep extra].
  - destruct (fresh b2) as [x b3] eqn:E3. cbv beta iota.
    destruct (fresh_spec a b2 x b3 E3) as (_ & _ & He & Ho);
      [ nexts; lia | ].
    exact (Hpre x b3 He Ho).
  - cbv zeta beta iota.
    pose proof (ext_note_simples a (exn_extra_simples ap) b2) as H3.
    destruct (note_simples_fields (exn_extra_simples ap) b2) as (N1 & _).
    set (b3 := note_simples _ b2) in *. clearbody b3.
    apply Hpre.
    + apply (ext_trans _ _ _ _ H3). apply ext_add_facts; [ | no_plain ].
      intros fct y H Hy. into_split H; into_close H Hy.
      match goal with
      | Hc : (?p, ?s) ∈ combine extra _ |- _ =>
        pose proof (H2o p (or_intror (in_combine_l _ _ _ _ Hc)))
      end.
      mono.
    + pose proof (H2o ep (or_introl eq_refl)). mono.
Qed.
(* ------------------------------------------------------------------------ *)
(* The traversal                                                            *)
(* ------------------------------------------------------------------------ *)

Lemma good_not_made (a : acc) (x : node) :
  uinv a -> not_made x -> good a x.
Proof.
  intros Hinv Hx. split; [ intros k -> ; destruct (Hx k eq_refl) | ].
  intros Hw. destruct (prot_made a x Hinv Hw) as (k & -> & _).
  exact (Hx k eq_refl).
Qed.

Lemma tstep_env (env env' : traverse_env) (a a' : acc) :
  (uinv a -> env_good env a -> env_good env' a) ->
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
  uinv a -> ∀ x, x ∈ handler_params h -> good a x.
Proof.
  intros Hinv x Hx. destruct h as [params md body]. cbn in Hx.
  apply in_map_iff in Hx. destruct Hx as [y [<- _]].
  apply good_not_made; [ exact Hinv | intros k; discriminate ].
Qed.

Lemma env_good_add_rec_conts (env : traverse_env)
    (handlers : list (continuation * continuation_handler)) (a : acc) :
  uinv a -> env_good env a -> env_good (add_rec_conts env handlers) a.
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
      * apply tstep_ext. intros _ _. apply ext_add_facts; [ | no_plain ].
        intros fct y H Hy. into_split H; into_close H Hy; ok_tac.
      * apply (tstep_env _ (code_env env id ret exn cd));
          [ | apply (IHc id (Code params mc ret exn md cbody) Hin) ].
        intros Hinvb _ k ps y Hk Hy.
        assert (Hin' : (id, cd) ∈ acc_code_deps b)
          by exact (FlambdaMapProofs.lookup_in _ _ _ Ecd).
        cbn [code_env env_conts] in Hk.
        destruct Hk as [Hk | [Hk | []]]; injection Hk as <- <-;
          apply (ui_code_deps b Hinvb id cd y Hin');
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
  - intros ap env a. apply tstep_traverse_apply.
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

(* [add_closure_dep], which makes the constructors of the field of
   unknown arity: of a witness of a code dep, or of a node with any
   source. *)
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
    ∧ (∀ fct, fct ∈ L -> no_read fct)
    ∧ (∀ c from, Fact_constructor c unknown from ∈ L ->
         (∃ id cd, lookup id cds = Some cd ∧ from ∈ uw cd)
         ∨ Fact_any_source from ∈ L).
Proof.
  intros Hle Hc Hmc. unfold add_closure_dep.
  destruct (lookup (cld_closure_code_id d) cds) as [cd | ] eqn:E.
  - cbn [add_facts acc_code_deps acc_next acc_facts].
    split; [ reflexivity | split; [ lia | ] ].
    eexists. split; [ reflexivity | split ].
    + intros fct x H Hx. into_split H; into_close H Hx;
        try exact Hc; exact (Hmc _ _ E).
    + split; [ no_read_tac | ].
      intros c' from H. left. exists (cld_closure_code_id d), cd.
      split; [ exact E | ].
      into_split H; try discriminate H; injection H as _ <-; left;
        reflexivity.
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
    + split; [ no_read_tac | ].
      intros c' from H. right.
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
    ∧ (∀ fct, fct ∈ L -> no_read fct)
    ∧ (∀ c from, Fact_constructor c unknown from ∈ L ->
         (∃ id cd, lookup id cds = Some cd ∧ from ∈ uw cd)
         ∨ Fact_any_source from ∈ L).
Proof.
  revert b. induction ds as [ | d ds IH]; intros b Hle Hds Hmc;
    cbn [List.fold_left].
  - split; [ reflexivity | split; [ lia | ] ].
    exists []. split; [ reflexivity | split; [ | split ] ];
      [ intros fct x [] | intros fct [] | intros c from [] ].
  - destruct (closure_dep_step a0 cds d b Hle (Hds d (or_introl eq_refl))
                Hmc) as (Hc1 & Hn1 & L1 & Hf1 & Hi1 & Hr1 & Hk1).
    set (b1 := add_closure_dep cds d b) in *. clearbody b1.
    destruct (IH b1) as (Hc2 & Hn2 & L2 & Hf2 & Hi2 & Hr2 & Hk2);
      [ lia | | | ].
    { intros d' Hd'. eapply ok_node_mono; [ exact Hn1 | ].
      exact (Hds d' (or_intror Hd')). }
    { intros id cd E. eapply ok_node_mono; [ exact Hn1 | ].
      exact (Hmc id cd E). }
    split; [ congruence | split; [ lia | ] ].
    exists (L2 ++ L1). split; [ rewrite Hf2, Hf1, app_assoc; reflexivity | ].
    split; [ | split ].
    + intros fct x H Hx. apply in_app_or in H as [H | H];
        [ exact (Hi2 _ _ H Hx) | ].
      eapply ok_node_mono; [ exact Hn2 | exact (Hi1 _ _ H Hx) ].
    + intros fct H. apply in_app_or in H as [H | H];
        [ exact (Hr2 _ H) | exact (Hr1 _ H) ].
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

(* What the delayed deps keep of the invariant: the facts that they add
   read nothing with the field of unknown arity and put nothing into a
   widget or a witness, and their constructors of the field of unknown
   arity hold a witness of a code dep or a node with any source. *)
Lemma resolve_unknown (a : acc) (unit_symbols : list symbol) :
  uinv a ->
  acc_code_deps (resolve_delayed_deps unit_symbols a) = acc_code_deps a
  ∧ (∀ fct, fct ∈ acc_facts a ->
       fct ∈ acc_facts (resolve_delayed_deps unit_symbols a))
  ∧ (∀ fct, fct ∈ acc_facts (resolve_delayed_deps unit_symbols a) ->
       fct ∈ acc_facts a
       ∨ (no_read fct ∧ ∀ x, x ∈ into_nodes fct -> ~ prot a x))
  ∧ (∀ b from,
       Fact_constructor b unknown from
       ∈ acc_facts (resolve_delayed_deps unit_symbols a) ->
       uwit a from
       ∨ Fact_any_source from
         ∈ acc_facts (resolve_delayed_deps unit_symbols a)).
Proof.
  intros Hinv. unfold resolve_delayed_deps. cbv zeta.
  assert (Hcd : ∀ id cd x, lookup id (acc_code_deps a) = Some cd ->
                  x ∈ code_dep_nodes cd -> ok_node a a x).
  { intros id cd x E Hx. apply ok_good.
    exact (ui_code_deps a Hinv id cd x (FlambdaMapProofs.lookup_in _ _ _ E)
             Hx). }
  set (b1 := add_facts _ a).
  assert (H1 : ext a a b1).
  { apply ext_add_facts.
    - intros fct y H Hy. apply in_flat_map in H. destruct H as [d [Hd H]].
      pose proof (ok_good a a _ (ui_apply_deps a Hinv d Hd)) as Hw.
      unfold apply_dep_facts, alias_for_caller in H.
      destruct (lookup (ad_apply_code_id d) (acc_code_deps a)) as [cd | ]
        eqn:E;
        destruct (ad_function_containing_apply_expr d);
        into_split H; into_close H Hy; try exact Hw; try ok_tac;
        apply (Hcd _ cd _ E); left; reflexivity.
    - intros fct H. apply in_flat_map in H. destruct H as [d [_ H]].
      unfold apply_dep_facts, alias_for_caller in H.
      destruct (lookup _ _); destruct (ad_function_containing_apply_expr d);
        into_split H; subst; plain_tac. }
  clearbody b1.
  destruct H1 as [Hn1 (L1 & Hf1 & Hi1 & Hp1) Hc1 Hap1 Hcl1].
  destruct (closure_deps_fold a (acc_code_deps a) (acc_closure_deps b1) b1)
    as (Hc2 & Hn2 & L2 & Hf2 & Hi2 & Hr2 & Hk2).
  { exact Hn1. }
  { intros d Hd. destruct (Hcl1 d Hd) as [Hd' | H]; [ | exact H ].
    apply ok_good. exact (ui_closure_deps a Hinv d Hd'). }
  { intros id cd E. eapply ok_node_mono; [ exact Hn1 | ].
    apply (Hcd id cd); [ exact E | left; reflexivity ]. }
  set (b2 := List.fold_left _ (acc_closure_deps b1) b1) in *.
  clearbody b2.
  set (L3 := map _ _).
  assert (Hi3 : ∀ fct x, fct ∈ L3 -> x ∈ into_nodes fct -> ok_node a b2 x).
  { intros fct x H Hx. unfold L3 in H. into_split H; into_close H Hx.
    ok_tac. }
  cbn [add_facts acc_code_deps acc_facts].
  split; [ congruence | split; [ | split ] ].
  - intros fct H. apply in_or_app. right. rewrite Hf2, Hf1.
    apply in_or_app. right. apply in_or_app. right. exact H.
  - intros fct H. rewrite Hf2, Hf1 in H.
    apply in_app_or in H as [H | H].
    { right. split; [ unfold L3 in H; into_split H; subst; plain_tac | ].
      intros x Hx. exact (ok_not_prot a _ _ Hinv (Hi3 _ _ H Hx)). }
    apply in_app_or in H as [H | H].
    { right. split; [ exact (Hr2 _ H) | ].
      intros x Hx. exact (ok_not_prot a _ _ Hinv (Hi2 _ _ H Hx)). }
    apply in_app_or in H as [H | H]; [ | left; exact H ].
    right. split.
    + specialize (Hp1 _ H). destruct fct; cbn [plain no_read] in *; auto.
    + intros x Hx. exact (ok_not_prot a _ _ Hinv (Hi1 _ _ H Hx)).
  - intros b from Hb. rewrite Hf2, Hf1 in Hb.
    apply in_app_or in Hb as [Hb | Hb].
    + exfalso. unfold L3 in Hb. into_split Hb. discriminate Hb.
    + apply in_app_or in Hb as [Hb | Hb].
      * destruct (Hk2 _ _ Hb) as [(id & cd & E & Hx) | H].
        -- left. exists id, cd.
           split; [ exact (FlambdaMapProofs.lookup_in _ _ _ E) | exact Hx ].
        -- right. apply in_or_app. right. rewrite Hf2. apply in_or_app.
           left. exact H.
      * apply in_app_or in Hb as [Hb | Hb].
        -- exfalso. specialize (Hp1 _ Hb). cbn [plain] in Hp1.
           exact (Hp1 eq_refl).
        -- left. exact (ui_constructor a Hinv _ _ Hb).
Qed.

Lemma uinv_empty : uinv empty_acc.
Proof. constructor; cbn; intros; repeat intro; firstorder. Qed.

(* The facts of the unit put into a widget only its read, and into a
   witness of unknown arity of a code dep only constructors of it and
   parameters of which it is the base; each such witness has the code id
   of its code; and their constructors of the field of unknown arity hold
   such a witness or a node with any source. *)
Lemma unit_unknown (flag : reaper_preserve_direct_calls) (u : unit_interface)
    (e : expr) :
  let a := traverse_unit_acc flag u e in
  (∀ x n fct,
     Fact_accessor x unknown n ∈ acc_facts a -> fct ∈ acc_facts a ->
     x ∈ into_nodes fct -> wid_fact fct n)
  ∧ (∀ id cd x fct,
       (id, cd) ∈ acc_code_deps a -> x ∈ uw cd -> fct ∈ acc_facts a ->
       x ∈ into_nodes fct -> wit_fact x fct)
  ∧ (∀ id cd x,
       (id, cd) ∈ acc_code_deps a -> x ∈ uw cd ->
       Fact_constructor x Field_code_id_of_call_witness (Node_code_id id)
       ∈ acc_facts a)
  ∧ (∀ b from,
       Fact_constructor b unknown from ∈ acc_facts a ->
       uwit a from ∨ Fact_any_source from ∈ acc_facts a).
Proof.
  unfold traverse_unit_acc. cbv zeta.
  set (a1 := add_fact (Fact_any_source Node_all_constants) empty_acc).
  assert (Hinv1 : uinv a1).
  { apply (ext_close empty_acc); [ exact uinv_empty | ].
    apply ext_add_fact; [ intros y [<- | []]; ok_tac | plain_tac ]. }
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
  assert (H4 : uinv a4 ∧ grows a3 a4).
  { apply (ext_close a3); [ exact Hinv3 | ].
    apply ext_add_fact; [ intros y [] | plain_tac ]. }
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
  assert (Hinv6 : uinv (add_fact (Fact_any_usage r) a5)).
  { apply (ext_close a5); [ exact Hinv5 | ].
    apply ext_add_fact; [ intros y [] | plain_tac ]. }
  set (a6 := add_fact (Fact_any_usage r) a5) in *. clearbody a6.
  destruct (resolve_unknown _ (static_names symbols_of_binding e) Hinv6)
    as (Hc & Hsub & Hnew & Hcons).
  set (a' := resolve_delayed_deps _ a6) in *. clearbody a'.
  assert (Hold : ∀ x n,
            Fact_accessor x unknown n ∈ acc_facts a' ->
            Fact_accessor x unknown n ∈ acc_facts a6).
  { intros x n H. destruct (Hnew _ H) as [H' | [H' _]]; [ exact H' | ].
    cbn [no_read] in H'. destruct (H' eq_refl). }
  split; [ | split; [ | split ] ].
  - intros x n fct Hw Hfct Hx. apply Hold in Hw.
    destruct (Hnew _ Hfct) as [H | [_ H]].
    + exact (ui_wid a6 Hinv6 _ _ _ Hw H Hx).
    + destruct (H x Hx (or_introl (ex_intro _ n Hw))).
  - intros id cd x fct Hin Hx Hfct Hxf. rewrite Hc in Hin.
    destruct (Hnew _ Hfct) as [H | [_ H]].
    + exact (ui_wit a6 Hinv6 _ _ _ _ Hin Hx H Hxf).
    + destruct (H x Hxf (or_intror (ex_intro _ id (ex_intro _ cd
                                                     (conj Hin Hx))))).
  - intros id cd x Hin Hx. rewrite Hc in Hin. apply Hsub.
    exact (ui_code_id a6 Hinv6 _ _ _ Hin Hx).
  - intros b from H. destruct (Hcons b from H) as [(id & cd & Hin & Hx) | H'].
    + left. exists id, cd. rewrite Hc. auto.
    + right. exact H'.
Qed.

(* One conjunct of [only_read_into] for a widget [x] whose facts are
   [wid_fact] ([Hok]), as in FlambdaReaperLeastProofs.AutoWidgets. *)
Ltac read_only_conj x Hok :=
  first
    [ lazymatch goal with
      | |- negb (mem_node x ?l) = true =>
        let Em := fresh "Em" in
        destruct (mem_node x l) eqn:Em; [ | reflexivity ];
        exfalso; apply mem_node_In in Em;
        cbn [f_any_source f_zero_alloc_source facts_of_list] in Em;
        first
          [ destruct Em
          | let fct := fresh "fct" in let Hf := fresh "Hf" in
            let Hy := fresh "Hy" in
            destruct (facts_of_list_in _ _ _ Em) as [fct [Hf Hy]];
            destruct fct; cbn in Hy; try contradiction;
            destruct Hy as [Hy | []]; subst;
            exact (Hok _ Hf (or_introl eq_refl)) ]
      end
    | let y := fresh "y" in let Hy := fresh "Hy" in
      let fct := fresh "fct" in let Hf := fresh "Hf" in
      let Hy' := fresh "Hy" in
      apply forallb_forall; intros y Hy;
      destruct (facts_of_list_in _ _ _ Hy) as [fct [Hf Hy']];
      destruct fct; cbn in Hy'; try contradiction;
      destruct Hy' as [<- | []]; cbv beta iota;
      repeat match goal with
      | |- context [node_eqb ?z x] =>
        let E := fresh "E" in
        destruct (node_eqb z x) eqn:E; [ apply node_eqb_eq in E; subst z | ]
      end; cbn [negb orb andb]; try reflexivity;
      let Hw := fresh "Hw" in
      pose proof (Hok _ Hf) as Hw; cbn [into_nodes wid_fact] in Hw;
      first
        [ solve [ destruct (Hw (or_introl eq_refl)) ]
        | solve [ destruct (Hw (or_intror (or_introl eq_refl))) ]
        | rewrite (Hw (or_introl eq_refl)), node_eqb_refl; reflexivity
        | destruct (Hw (or_introl eq_refl)) as [-> ->];
          rewrite field_eqb_refl, node_eqb_refl; reflexivity ] ].

(* Each node read with the field of unknown arity is only read into. *)
Lemma unknown_read_only (flag : reaper_preserve_direct_calls)
    (u : unit_interface) (e : expr) (x n : node) :
  (x, unknown, n) ∈ f_accessor (traverse_unit flag u e) ->
  only_read_into (traverse_unit flag u e) x unknown n true = true.
Proof.
  destruct (unit_unknown flag u e) as (Hw & _).
  rewrite traverse_unit_acc_facts.
  set (a := traverse_unit_acc flag u e) in *. clearbody a.
  intros Hin.
  destruct (facts_of_list_in _ _ _ Hin) as [fct [Hfct H]].
  destruct fct; cbn in H; try contradiction.
  destruct H as [Heq | []]. injection Heq as -> -> ->.
  assert (Hok : ∀ fct', fct' ∈ acc_facts a -> x ∈ into_nodes fct' ->
                  wid_fact fct' n)
    by (intros fct' H1 H2; exact (Hw _ _ _ Hfct H1 H2)).
  unfold only_read_into.
  repeat (apply andb_true_intro; split).
  all: read_only_conj x Hok.
Qed.

End UnknownArity.

(* ======================================================================== *)
(* [check_unknown_widgets]                                                  *)
(* ======================================================================== *)

Abbreviation unknown := (Field_call_witness Unknown_arity_code_pointer).

Ltac unfold_rels :=
  cbn [graph_of_facts solution_of_tables holds g_alias g_use g_accessor
       g_constructor g_argument g_parameter g_propagate
       g_alias_if_any_source g_any_usage g_any_source g_any_source_if_used
       g_zero_alloc_source s_alias s_any_usage s_any_source s_has_usage
       s_has_source s_usages s_sources s_reading_field s_escaping_field
       s_field_used] in *.

(* [read_of], [widget_any_source] and [widget_sources] of
   FlambdaReaperLeastProofs.v, for any field that is not local. *)
Section Widgets.

Variable is_local : field -> bool.
Variable F : facts.
Variable T : tables.
Variable f : field.
Hypothesis Hleast : least_solution is_local F T.
Hypothesis Hf : is_local f = false.

Lemma read_of_any (a n y from : node) :
  FlambdaReaperLeastProofs.read_only F a f n true ->
  (y, f, n) ∈ f_accessor F ->
  mem_pair a from (t_alias T) = true ->
  mem_node n (t_any_source T) = true ∨ mem_pair y from (t_alias T) = true.
Proof.
  intros Hro Hacc Hal.
  pose proof (FlambdaReaperLeastProofs.least_closed is_local F T Hleast)
    as Hc.
  apply (FlambdaReaperLeastProofs.least_derivable is_local F T Hleast
           (Atom_alias a from)) in Hal.
  destruct (FlambdaReaperLeastProofs.alias_into is_local F T a _ n true Hro
              Hf from Hal)
    as [(b & Hb & Hu & Hg) | [(bs & Hs & Hg & Hb) | [_ Has]]].
  - right. apply (FlambdaReaperLeastProofs.least_holds is_local F T Hleast)
      in Hu.
    unfold_rels.
    exact (alias_constructor_usage _ _ _ Hc b n _ y from Hb Hu Hg Hacc).
  - right. apply (FlambdaReaperLeastProofs.least_holds is_local F T Hleast)
      in Hs.
    unfold_rels.
    exact (alias_accessor_source _ _ _ Hc n bs _ y from Hacc Hs Hg Hb).
  - left. exact (FlambdaReaperLeastProofs.least_holds is_local F T Hleast _
                   Has).
Qed.

Lemma widget_any_source_any (a a' n : node) :
  FlambdaReaperLeastProofs.read_only F a f n true ->
  (a', f, n) ∈ f_accessor F ->
  mem_node a (t_any_source T) = true -> mem_node a' (t_any_source T) = true.
Proof.
  intros Hro Hacc' Has.
  pose proof (FlambdaReaperLeastProofs.least_closed is_local F T Hleast)
    as Hc.
  apply (FlambdaReaperLeastProofs.least_derivable is_local F T Hleast
           (Atom_any_source a)) in Has.
  destruct (FlambdaReaperLeastProofs.any_source_into is_local F T a _ n true
              Hro Has)
    as [(from & Hal & Hfrom) | Hn].
  - apply (FlambdaReaperLeastProofs.least_holds is_local F T Hleast)
      in Hal, Hfrom.
    destruct (read_of_any a n a' from Hro Hacc' Hal) as [Hn | Hal'].
    + exact (any_source_accessor _ _ _ Hc n _ a' Hn Hacc' Hf).
    + exact (any_source_alias _ _ _ Hc from a' Hal' Hfrom).
  - apply (FlambdaReaperLeastProofs.least_holds is_local F T Hleast) in Hn.
    unfold_rels.
    exact (any_source_accessor _ _ _ Hc n _ a' Hn Hacc' Hf).
Qed.

Lemma widget_sources_any (a a' n y : node) :
  FlambdaReaperLeastProofs.read_only F a f n true ->
  FlambdaReaperLeastProofs.read_only F a' f n true ->
  (a, f, n) ∈ f_accessor F -> (a', f, n) ∈ f_accessor F ->
  mem_pair a y (t_sources T) = true -> mem_pair a' y (t_sources T) = true.
Proof.
  intros Hro Hro' Hacc Hacc' Hs.
  pose proof (FlambdaReaperLeastProofs.least_closed is_local F T Hleast)
    as Hc.
  apply (FlambdaReaperLeastProofs.least_derivable is_local F T Hleast
           (Atom_sources a y)) in Hs.
  destruct (FlambdaReaperLeastProofs.sources_into is_local F T a _ n true Hro
              y Hs)
    as (from & Hfrom & Hgf & Hal & Hga).
  apply (FlambdaReaperLeastProofs.least_holds is_local F T Hleast)
    in Hfrom, Hal.
  unfold_rels.
  destruct (read_of_any a n a' from Hro Hacc' Hal) as [Hn | Hal'].
  - pose proof (any_source_accessor _ _ _ Hc n _ a Hn Hacc Hf).
    unfold_rels. congruence.
  - assert (Hga' : mem_node a' (t_any_source T) = false).
    { destruct (mem_node a' (t_any_source T)) eqn:E; [ | reflexivity ].
      rewrite (widget_any_source_any a' a n Hro' Hacc E) in Hga.
      discriminate. }
    exact (sources_alias _ _ _ Hc from a' y Hfrom Hgf Hal' Hga').
Qed.

End Widgets.

Lemma unknown_not_local (is_local : field -> bool) :
  is_local_slots_only is_local -> is_local unknown = false.
Proof.
  intros Hloc. destruct (is_local unknown) eqn:E; [ | reflexivity ].
  destruct (Hloc _ E) as [(w & Hw) | (fs & Hfs)]; discriminate.
Qed.

Theorem traverse_unknown_widgets : traverse_unknown_widgets_statement.
Proof.
  intros flag is_local u e T Hslots Hleast.
  pose proof (unknown_not_local is_local Hslots) as Hu.
  unfold check_unknown_widgets. apply forallb_forall.
  intros [[a f] n] Hacc. apply forallb_forall.
  intros [[a' f'] n'] Hacc'. cbn beta iota.
  destruct (field_eqb f unknown && field_eqb f' f && node_eqb n' n) eqn:E;
    [ | reflexivity ].
  apply andb_true_iff in E as [E Hn]. apply andb_true_iff in E as [Hf Hf'].
  apply field_eqb_eq in Hf, Hf'. apply node_eqb_eq in Hn. subst.
  pose proof (FlambdaReaperLeastProofs.only_read_into_read_only _ _ _ _ _
                (UnknownArity.unknown_read_only flag u e _ _ Hacc)) as Hro.
  pose proof (FlambdaReaperLeastProofs.only_read_into_read_only _ _ _ _ _
                (UnknownArity.unknown_read_only flag u e _ _ Hacc')) as Hro'.
  cbn [negb orb]. apply andb_true_iff. split.
  - apply Bool.eqb_true_iff.
    destruct (mem_node a (t_any_source T)) eqn:Ea,
      (mem_node a' (t_any_source T)) eqn:Ea'; try reflexivity.
    + rewrite (widget_any_source_any is_local _ T _ Hleast Hu a a' n Hro
                 Hacc' Ea) in Ea'.
      discriminate.
    + rewrite (widget_any_source_any is_local _ T _ Hleast Hu a' a n Hro'
                 Hacc Ea') in Ea.
      discriminate.
  - apply forallb_forall. intros [x y] Hin. cbn beta iota.
    destruct (node_eqb x a) eqn:Ex; [ | reflexivity ].
    apply node_eqb_eq in Ex. subst x. cbn.
    exact (widget_sources_any is_local _ T _ Hleast Hu a a' n y Hro Hro'
             Hacc Hacc' (proj2 (mem_pair_In _ _ _) Hin)).
Qed.

(* ======================================================================== *)
(* [check_unknown_witnesses]                                                *)
(* ======================================================================== *)

Lemma constructor_of (l : list fact) (base from : node) (f : field) :
  Fact_constructor base f from ∈ l ->
  (base, f, from) ∈ f_constructor (facts_of_list l).
Proof.
  intros H. cbn [f_constructor facts_of_list]. apply in_flat_map.
  exists (Fact_constructor base f from). split; [ exact H | left; reflexivity ].
Qed.

Section Witnesses.

Variable a : acc.
Variable is_local : field -> bool.
Variable T : tables.
Hypothesis Hloc : is_local_slots_only is_local.
Hypothesis Hleast : least_solution is_local (facts_of_list (acc_facts a)) T.
Hypothesis Hwit : ∀ id cd x fct,
  (id, cd) ∈ acc_code_deps a -> x ∈ cd_unknown_arity_call_witnesses cd ->
  fct ∈ acc_facts a ->
  x ∈ FlambdaReaperLeastProofs.AutoWidgets.into_nodes fct ->
  UnknownArity.wit_fact x fct.
Hypothesis Hcid : ∀ id cd x,
  (id, cd) ∈ acc_code_deps a -> x ∈ cd_unknown_arity_call_witnesses cd ->
  Fact_constructor x Field_code_id_of_call_witness (Node_code_id id)
  ∈ acc_facts a.
Hypothesis Hcons : ∀ b from,
  Fact_constructor b unknown from ∈ acc_facts a ->
  UnknownArity.uwit a from ∨ Fact_any_source from ∈ acc_facts a.

Abbreviation F := (facts_of_list (acc_facts a)).
Abbreviation G := (graph_of_facts F).
Abbreviation S := (solution_of_tables T).
Abbreviation D := (derivable is_local G S).

Hypothesis Hro : ∀ x n,
  (x, unknown, n) ∈ f_accessor F ->
  FlambdaReaperLeastProofs.read_only F x unknown n true.

(* Nothing is aliased into a witness of unknown arity of a code dep. *)
Lemma no_alias_into_witness (id : code_id) (cd : code_dep) (x from : node) :
  (id, cd) ∈ acc_code_deps a -> x ∈ cd_unknown_arity_call_witnesses cd ->
  ~ D (Atom_alias x from).
Proof.
  intros Hin Hx Dal. pose proof (fun fct => Hwit id cd x fct Hin Hx) as Hk.
  inversion Dal; subst;
    repeat match goal with
    | H : derivable _ _ _ (Atom_reading_field _ _) |- _ =>
      inversion H; subst; clear H
    end;
    unfold_rels;
    match goal with
    | H : (_, _) ∈ f_alias _ |- _ =>
      exact (Hk _ (FlambdaReaperWrapperAutoProofs.alias_in _ _ _ H)
               (or_introl eq_refl))
    | H : (_, _, _) ∈ f_propagate _ |- _ =>
      exact (Hk _ (FlambdaReaperWrapperAutoProofs.propagate_in _ _ _ _ H)
               (or_introl eq_refl))
    | H : (_, _, _) ∈ f_alias_if_any_source _ |- _ =>
      exact (Hk _ (FlambdaReaperWrapperAutoProofs.if_any_source_in _ _ _ _ H)
               (or_introl eq_refl))
    | H : (_, _, _) ∈ f_parameter _ |- _ =>
      exact (Hk _ (FlambdaReaperWrapperAutoProofs.parameter_in _ _ _ _ H)
               (or_intror (or_introl eq_refl)) eq_refl)
    | H : (_, _, _) ∈ f_accessor _ |- _ =>
      exact (Hk _ (FlambdaReaperWrapperAutoProofs.accessor_in _ _ _ _ H)
               (or_introl eq_refl))
    end.
Qed.

(* A source of a node read with the field of unknown arity is held in that
   field of a node and is a witness of unknown arity of a code dep. *)
Lemma unknown_source (x n s : node) :
  (x, unknown, n) ∈ f_accessor F -> (x, s) ∈ t_sources T ->
  (∃ b, (b, unknown, s) ∈ f_constructor F) ∧ UnknownArity.uwit a s.
Proof.
  intros Hacc Hs.
  pose proof (FlambdaReaperLeastProofs.least_closed is_local F T Hleast)
    as Hc.
  pose proof (unknown_not_local is_local Hloc) as Hu.
  pose proof (Hro x n Hacc) as Hrx.
  apply (proj2 (mem_pair_In _ _ _)) in Hs.
  apply (FlambdaReaperLeastProofs.least_derivable is_local F T Hleast
           (Atom_sources x s)) in Hs.
  destruct (FlambdaReaperLeastProofs.sources_into is_local F T x unknown n
              true Hrx s Hs) as (from & Dfs & Hgf & Dal & Hgx).
  assert (Hfrom : ∃ b, (b, unknown, from) ∈ f_constructor F).
  { destruct (FlambdaReaperLeastProofs.alias_into is_local F T x unknown n
                true Hrx Hu from Dal)
      as [(b & Hb & _) | [(bs & _ & _ & Hb) | [_ Dn]]].
    - exists b. exact Hb.
    - exists bs. exact Hb.
    - exfalso.
      apply (FlambdaReaperLeastProofs.least_holds is_local F T Hleast) in Dn.
      pose proof (any_source_accessor _ _ _ Hc n _ x Dn Hacc Hu).
      unfold_rels. congruence. }
  destruct Hfrom as (b & Hb).
  destruct (Hcons _ _ (FlambdaReaperWrapperAutoProofs.constructor_in _ _ _ _
                         Hb))
    as [(id & cd & Hin & Hx) | Hany].
  - assert (Hsf : s = from).
    { inversion Dfs; subst; [ reflexivity | reflexivity | ].
      exfalso. eapply no_alias_into_witness; eassumption. }
    subst s. split; [ exists b; exact Hb | ].
    exists id, cd. auto.
  - exfalso.
    pose proof (any_source_input _ _ _ Hc from
                  (FlambdaReaperWrapperAutoProofs.any_source_of _ _ Hany))
      as H.
    unfold_rels. congruence.
Qed.

Lemma unknown_witnesses : check_unknown_witnesses F T = true.
Proof.
  unfold check_unknown_witnesses. apply forallb_forall.
  intros [[x f] n] Hacc. cbv beta iota.
  destruct (field_eqb f unknown) eqn:Ef; [ | reflexivity ].
  apply field_eqb_eq in Ef. subst f. cbn [negb orb].
  apply forallb_forall. intros [x' s] Hs. cbv beta iota.
  destruct (node_eqb x' x) eqn:Ex; [ | reflexivity ].
  apply node_eqb_eq in Ex. subst x'.
  destruct (mem_node s (t_has_source T)); [ | reflexivity ].
  cbn [negb orb].
  destruct (unknown_source x n s Hacc Hs) as [(b & Hb) (id & cd & Hin & Hx)].
  apply andb_true_intro. split.
  - apply existsb_exists. exists (b, unknown, s). split; [ exact Hb | ].
    rewrite field_eqb_refl, node_eqb_refl. reflexivity.
  - apply existsb_exists.
    exists (s, Field_code_id_of_call_witness, Node_code_id id).
    split; [ exact (constructor_of _ _ _ _ (Hcid _ _ _ Hin Hx)) | ].
    rewrite field_eqb_refl, node_eqb_refl. reflexivity.
Qed.

End Witnesses.

Theorem traverse_unknown_witnesses : traverse_unknown_witnesses_statement.
Proof.
  intros flag is_local u e T Hloc Hleast.
  destruct (UnknownArity.unit_unknown flag u e)
    as (_ & Hwit & Hcid & Hcons).
  pose proof (fun x n H =>
                FlambdaReaperLeastProofs.only_read_into_read_only _ _ _ _ _
                  (UnknownArity.unknown_read_only flag u e x n H)) as Hro.
  rewrite traverse_unit_acc_facts in Hleast, Hro |- *.
  exact (unknown_witnesses _ is_local T Hloc Hleast Hwit Hcid Hcons Hro).
Qed.

(* ======================================================================== *)
(* The statement                                                            *)
(* ======================================================================== *)

Theorem indirect_arm_needs_no_source_auto_end_to_end :
  indirect_arm_needs_no_source_auto_end_to_end_statement.
Proof.
  intros is_local u e T max_unbox_size lambda_lifting to_unbox F code_deps
    cannot_change unboxed Hloc Hslots Hfrag Hleast Hunbox Hcount Hkinds.
  exact (FlambdaReaperWrapperShapeProofs
           .indirect_arm_needs_no_source_traversal
           Auto is_local u e T cannot_change max_unbox_size lambda_lifting
           (derive_dominated T) to_unbox Hloc Hslots Hfrag Hleast
           (proj1 (FlambdaReaperCallingConventionsProofs
                     .derive_cannot_change_least F T))
           (FlambdaReaperWrapperAutoProofs.derive_dominated_checked T) Hunbox
           (FlambdaReaperLeastProofs.least_apply_widgets is_local F T Hloc
              Hleast (FlambdaReaperLeastProofs.traverse_widgets_read_only u e))
           (FlambdaReaperWrapperAutoProofs.traverse_known_witnesses is_local u
              e T Hloc Hleast)
           (traverse_unknown_widgets Auto is_local u e T Hloc Hleast)
           (traverse_unknown_witnesses Auto is_local u e T Hloc Hleast)
           Hcount Hkinds).
Qed.

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

(* The proofs of FlambdaReaperIndirectDeadFramesTraversalSpec.v. Nothing
   here needs review.

   [chain S L x n]: what [check_unknown_chains] asks of the read [x] off
   [n], on the list [L] ([check_of_chains]), the nodes it looks at being
   of [S]. No fact of a list that has no node of [S] breaks it
   ([chain_frame]).

   [uinv S a]: the nodes of the facts of [a] are made before it, those of
   [S] too, each read of the field of unknown arity has [chain S], and
   the deps have no node of [S]. Each step of the traversal but the call
   of unknown arity adds facts of nodes not of [S] and no such read
   ([ext], as in FlambdaReaperLeastProofs.AutoWidgets, on all the nodes of
   a fact). The call of unknown arity ([uinv_call_unknown]) makes its
   chain of fresh nodes ([deps_spec]), whose facts check it alone; [S]
   grows by the nodes it makes. Nothing is asked of the unit or of the
   flag. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaWellFormedSpec FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperSolution FlambdaReaperSolutionProofs
  FlambdaReaperCallingConventions FlambdaReaperLeastSpec
  FlambdaReaperIndirectDeadFramesSpec
  FlambdaReaperIndirectDeadFramesTraversalSpec.
From OxCamlFlambda2 Require FlambdaMapProofs FlambdaReaperFlowProofs
  FlambdaReaperLeastProofs FlambdaReaperIndirectDeadFramesProofs.
Import ListNotations.

Module Chains.

Module AW := FlambdaReaperLeastProofs.AutoWidgets.

Abbreviation unknown := (Field_call_witness Unknown_arity_code_pointer).
Abbreviation first_result := (Field_return_of_call (Normal 0)).
Abbreviation cid := Field_code_id_of_call_witness.
Abbreviation env_conts := FlambdaReaperTraversal.env_conts.
Abbreviation into_nodes := AW.into_nodes.
Abbreviation made_lt := AW.made_lt.
Abbreviation not_made := AW.not_made.

(* All the nodes of a fact. *)
Definition nodes_of (fct : fact) : list node :=
  match fct with
  | Fact_alias x y | Fact_use x y | Fact_any_source_if_used x y
  | Fact_accessor x _ y | Fact_constructor x _ y | Fact_argument x _ y
  | Fact_parameter x _ y => [x; y]
  | Fact_propagate c x y | Fact_alias_if_any_source c x y => [c; x; y]
  | Fact_any_usage x | Fact_any_source x => [x]
  end.

Lemma into_in_nodes (fct : fact) (x : node) :
  x ∈ into_nodes fct -> x ∈ nodes_of fct.
Proof.
  destruct fct; cbn; intros H; repeat destruct H as [<- | H]; cbn; auto;
    contradiction.
Qed.

(* ------------------------------------------------------------------------ *)
(* The check, on a list of facts                                            *)
(* ------------------------------------------------------------------------ *)

(* [fct] is of [L], and is all that puts something into [x]. *)
Definition uniq_into (L : list fact) (x : node) (fct : fact) : Prop :=
  fct ∈ L ∧ ∀ fct', fct' ∈ L -> x ∈ into_nodes fct' -> fct' = fct.

(* Nothing but aliases and propagations from [x] puts anything into
   [w]. *)
Definition alias_only (L : list fact) (w x : node) : Prop :=
  ∀ fct, fct ∈ L -> w ∈ into_nodes fct ->
    fct = Fact_alias w x ∨ ∃ c, fct = Fact_propagate c w x.

Definition aliased_from (L : list fact) (w x : node) : Prop :=
  Fact_alias w x ∈ L ∨ ∃ c, Fact_propagate c w x ∈ L.

(* What [check_unknown_chains] asks of a read [x] off [n], the nodes it
   looks at being of [S]. *)
Definition chain (S : node -> Prop) (L : list fact) (x n : node) : Prop :=
  S x ∧ uniq_into L x (Fact_accessor x unknown n)
  ∧ ((∀ c, Fact_accessor c cid x ∉ L)
     ∨ S n ∧ ∃ w, uniq_into L n (Fact_accessor n first_result w))
  ∧ ∀ w, aliased_from L w x -> S w ∧ alias_only L w x.

Definition chains (S : node -> Prop) (L : list fact) : Prop :=
  ∀ x n, Fact_accessor x unknown n ∈ L -> chain S L x n.

(* No fact of [L] has a node of [S]. *)
Definition avoids (S : node -> Prop) (L : list fact) : Prop :=
  ∀ fct y, fct ∈ L -> y ∈ nodes_of fct -> ~ S y.

Lemma avoids_into (S : node -> Prop) (L : list fact) (fct : fact)
    (y : node) :
  avoids S L -> fct ∈ L -> y ∈ into_nodes fct -> ~ S y.
Proof. intros H Hf Hy. exact (H _ _ Hf (into_in_nodes _ _ Hy)). Qed.

Lemma chain_frame (S S' : node -> Prop) (L1 L2 : list fact) (x n : node) :
  (∀ y, S y -> S' y) -> chain S L1 x n -> avoids S L2 ->
  chain S' (L1 ++ L2) x n.
Proof.
  intros HS (Hx & [Hr Hu] & Hcid & Hal) Hav.
  split; [ exact (HS _ Hx) | ].
  split; [ split; [ apply in_or_app; left; exact Hr | ] | split ].
  - intros fct Hf Hy.
    apply in_app_or in Hf as [Hf | Hf]; [ exact (Hu _ Hf Hy) | ].
    destruct (avoids_into _ _ _ _ Hav Hf Hy Hx).
  - destruct Hcid as [Hno | (Hn & w & Hw1 & Hw2)].
    + left. intros c Hc. apply in_app_or in Hc as [Hc | Hc];
        [ exact (Hno c Hc) | ].
      exact (Hav _ x Hc (or_intror (or_introl eq_refl)) Hx).
    + right. split; [ exact (HS _ Hn) | ]. exists w.
      split; [ apply in_or_app; left; exact Hw1 | ].
      intros fct Hf Hy. apply in_app_or in Hf as [Hf | Hf];
        [ exact (Hw2 _ Hf Hy) | ].
      destruct (avoids_into _ _ _ _ Hav Hf Hy Hn).
  - intros w Hw.
    assert (Hw1 : aliased_from L1 w x).
    { destruct Hw as [Hw | (c & Hw)].
      - apply in_app_or in Hw. destruct Hw as [Hw | Hw]; [ left; exact Hw | ].
        exfalso. apply (Hav _ x Hw); [ cbn; auto | exact Hx ].
      - apply in_app_or in Hw. destruct Hw as [Hw | Hw];
          [ right; exists c; exact Hw | ].
        exfalso. apply (Hav _ x Hw); [ cbn; auto | exact Hx ]. }
    destruct (Hal w Hw1) as [Hsw Hao]. split; [ exact (HS _ Hsw) | ].
    intros fct Hf Hy. apply in_app_or in Hf as [Hf | Hf];
      [ exact (Hao _ Hf Hy) | ].
    destruct (avoids_into _ _ _ _ Hav Hf Hy Hsw).
Qed.

(* [chain] reads [L] as a set. *)
Lemma chain_same (S : node -> Prop) (L L' : list fact) (x n : node) :
  (∀ fct, fct ∈ L <-> fct ∈ L') -> chain S L x n -> chain S L' x n.
Proof.
  intros He (Hx & [Hr Hu] & Hcid & Hal).
  split; [ exact Hx | ].
  split; [ split; [ apply He; exact Hr | ] | split ].
  - intros fct Hf Hy. apply He in Hf. exact (Hu _ Hf Hy).
  - destruct Hcid as [Hno | (Hn & w & Hw1 & Hw2)].
    + left. intros c Hc. apply He in Hc. exact (Hno c Hc).
    + right. split; [ exact Hn | ]. exists w.
      split; [ apply He; exact Hw1 | ].
      intros fct Hf Hy. apply He in Hf. exact (Hw2 _ Hf Hy).
  - intros w Hw.
    assert (Hw1 : aliased_from L w x).
    { destruct Hw as [Hw | (c & Hw)]; apply He in Hw;
        [ left; exact Hw | right; exists c; exact Hw ]. }
    destruct (Hal w Hw1) as [Hsw Hao]. split; [ exact Hsw | ].
    intros fct Hf Hy. apply He in Hf. exact (Hao _ Hf Hy).
Qed.

Lemma chain_frame_l (S S' : node -> Prop) (L1 L2 : list fact) (x n : node) :
  (∀ y, S y -> S' y) -> chain S L1 x n -> avoids S L2 ->
  chain S' (L2 ++ L1) x n.
Proof.
  intros HS H Hav. apply (chain_same _ (L1 ++ L2)).
  - intros fct. split; intros Hf; apply in_app_or in Hf as [Hf | Hf];
      apply in_or_app; auto.
  - exact (chain_frame _ _ _ _ _ _ HS H Hav).
Qed.

(* ------------------------------------------------------------------------ *)
(* From the lists to the check                                              *)
(* ------------------------------------------------------------------------ *)

Lemma flat_in {B : Type} (g : fact -> list B) (l : list fact) (y : B) :
  y ∈ flat_map g l -> ∃ fct, fct ∈ l ∧ y ∈ g fct.
Proof. intros H. apply in_flat_map in H. exact H. Qed.

Lemma flat_of {B : Type} (g : fact -> list B) (l : list fact) (y : B)
    (fct : fact) :
  fct ∈ l -> y ∈ g fct -> y ∈ flat_map g l.
Proof. intros H1 H2. apply in_flat_map. exists fct. auto. Qed.

(* One conjunct of [only_read_into] or [only_alias_into] at [x], where
   [Hok] says what the facts into [x] are. *)
Ltac into_conj x Hok :=
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
            destruct (flat_in _ _ _ Em) as [fct [Hf Hy]];
            destruct fct; cbn in Hy; try contradiction;
            destruct Hy as [Hy | []]; subst;
            let Hw := fresh "Hw" in
            pose proof (Hok _ Hf (or_introl eq_refl)) as Hw;
            repeat match type of Hw with
            | _ ∨ _ => destruct Hw as [Hw | Hw]
            | ∃ _, _ => destruct Hw as [? Hw]
            end; discriminate ]
      end
    | let y := fresh "y" in let Hy := fresh "Hy" in
      let fct := fresh "fct" in let Hf := fresh "Hf" in
      let Hy' := fresh "Hy" in
      apply forallb_forall; intros y Hy;
      destruct (flat_in _ _ _ Hy) as [fct [Hf Hy']];
      destruct fct; cbn in Hy'; try contradiction;
      destruct Hy' as [<- | []]; cbv beta iota;
      repeat match goal with
      | |- context [node_eqb ?z x] =>
        let E := fresh "E" in
        destruct (node_eqb z x) eqn:E;
        [ apply node_eqb_eq in E; try subst z | ]
      end; cbn [negb orb andb]; try reflexivity;
      let Hw := fresh "Hw" in
      pose proof (Hok _ Hf) as Hw; cbn [into_nodes] in Hw;
      first
        [ specialize (Hw (or_introl eq_refl))
        | specialize (Hw (or_intror (or_introl eq_refl))) ];
      repeat match type of Hw with
      | _ ∨ _ => destruct Hw as [Hw | Hw]
      | ∃ _, _ => destruct Hw as [? Hw]
      end;
      try discriminate; injection Hw; intros; subst;
      rewrite ?field_eqb_refl, ?node_eqb_refl; reflexivity ].

Lemma only_read_into_of (L : list fact) (x : node) (f : field)
    (base : node) :
  (∀ fct, fct ∈ L -> x ∈ into_nodes fct -> fct = Fact_accessor x f base) ->
  only_read_into (facts_of_list L) x f base false = true.
Proof.
  intros Hok. unfold only_read_into.
  repeat (apply andb_true_intro; split).
  all: into_conj x Hok.
Qed.

Lemma only_alias_into_of (L : list fact) (w x : node) :
  alias_only L w x -> only_alias_into (facts_of_list L) w x = true.
Proof.
  intros Hok. unfold only_alias_into.
  repeat (apply andb_true_intro; split).
  all: into_conj w Hok.
Qed.

Lemma check_of_chains (S : node -> Prop) (L : list fact) :
  chains S L -> check_unknown_chains (facts_of_list L) = true.
Proof.
  intros H. unfold check_unknown_chains. apply forallb_forall.
  intros [[x f] n] Hin. cbv beta iota.
  destruct (flat_in _ _ _ Hin) as [fct [Hf Hy]].
  destruct fct; cbn in Hy; try contradiction.
  destruct Hy as [Heq | []]. injection Heq as -> -> ->.
  unfold field_eqb at 1.
  destruct (field_eq_dec f unknown) as [-> | ]; [ | reflexivity ].
  cbn [negb orb].
  destruct (H x n Hf) as (_ & [_ Hu] & Hcid & Hal).
  apply andb_true_intro; split; [ apply andb_true_intro; split;
    [ apply andb_true_intro; split | ] | ].
  - apply only_read_into_of. exact Hu.
  - destruct Hcid as [Hno | (_ & w & Hw1 & Hw2)].
    + apply orb_true_intro. left. apply negb_true_iff.
      destruct (existsb _ _) eqn:E; [ | reflexivity ].
      apply existsb_exists in E as [[[c f'] b] [Hc Hb]].
      apply andb_true_iff in Hb as [Hf' Hb].
      apply field_eqb_eq in Hf'. apply node_eqb_eq in Hb. subst f' b.
      destruct (flat_in _ _ _ Hc) as [fct [Hf2 Hy]].
      destruct fct; cbn in Hy; try contradiction.
      destruct Hy as [Heq | []]. injection Heq as -> -> ->.
      destruct (Hno c Hf2).
    + apply orb_true_intro. right. apply existsb_exists.
      exists (n, first_result, w). split.
      * apply (flat_of _ _ _ (Fact_accessor n first_result w)); [ exact Hw1 | ].
        left. reflexivity.
      * rewrite node_eqb_refl, field_eqb_refl. cbn [andb].
        apply only_read_into_of. exact Hw2.
  - apply forallb_forall. intros [w y] Hw. cbv beta iota.
    destruct (node_eqb y x) eqn:E; [ | reflexivity ].
    apply node_eqb_eq in E. subst y. cbn [negb orb].
    destruct (flat_in _ _ _ Hw) as [fct [Hf2 Hy]].
    destruct fct; cbn in Hy; try contradiction.
    destruct Hy as [Heq | []]. injection Heq as -> ->.
    apply only_alias_into_of. apply (Hal w). left. exact Hf2.
  - apply forallb_forall. intros [[c w] y] Hw. cbv beta iota.
    destruct (node_eqb y x) eqn:E; [ | reflexivity ].
    apply node_eqb_eq in E. subst y. cbn [negb orb].
    destruct (flat_in _ _ _ Hw) as [fct [Hf2 Hy]].
    destruct fct; cbn in Hy; try contradiction.
    destruct Hy as [Heq | []]. injection Heq as -> -> ->.
    apply only_alias_into_of. apply (Hal w). right. exists c. exact Hf2.
Qed.



(* ------------------------------------------------------------------------ *)
(* The invariant                                                            *)
(* ------------------------------------------------------------------------ *)

(* The nodes of a code dep and of an apply dep. *)
Definition code_dep_nodes (cd : code_dep) : list node :=
  cd_my_closure cd :: cd_exn cd :: cd_known_arity_call_witness cd
  :: cd_params cd ++ cd_return cd ++ cd_unknown_arity_call_witnesses cd.

Lemma in_cd_my (cd : code_dep) : cd_my_closure cd ∈ code_dep_nodes cd.
Proof. left. reflexivity. Qed.

Lemma in_cd_known (cd : code_dep) :
  cd_known_arity_call_witness cd ∈ code_dep_nodes cd.
Proof. right; right; left. reflexivity. Qed.

Lemma in_cd_unknown (cd : code_dep) (x : node) :
  x ∈ cd_unknown_arity_call_witnesses cd -> x ∈ code_dep_nodes cd.
Proof.
  intros H. right; right; right. apply in_or_app. right. apply in_or_app.
  right. exact H.
Qed.

Definition apply_dep_nodes (d : apply_dep) : list node :=
  [ad_apply_closure d; ad_apply_call_witness d].

Section Ext.

(* The nodes of the chains made so far. *)
Variable S : node -> Prop.

(* [x] is made before [a] and is not of [S]. *)
Definition good (a : acc) (x : node) : Prop :=
  made_lt x (acc_next a) ∧ ~ S x.

(* [x] is not made, or good in [a0], or made from [a0] to [a]. *)
Definition ok_node (a0 a : acc) (x : node) : Prop :=
  not_made x ∨ good a0 x
  ∨ ∃ k, x = Node_made k ∧ (acc_next a0 <= k)%positive
         ∧ (k < acc_next a)%positive.

Record uinv (a : acc) : Prop := {
  ui_nodes : ∀ fct x,
    fct ∈ acc_facts a -> x ∈ nodes_of fct -> made_lt x (acc_next a);
  ui_special : ∀ x, S x -> ∃ k, x = Node_made k ∧ (k < acc_next a)%positive;
  ui_chains : chains S (acc_facts a);
  ui_code_deps : ∀ id cd x,
    (id, cd) ∈ acc_code_deps a -> x ∈ code_dep_nodes cd -> good a x;
  ui_apply_deps : ∀ d x,
    d ∈ acc_apply_deps a -> x ∈ apply_dep_nodes d -> good a x;
  ui_closure_deps : ∀ d,
    d ∈ acc_closure_deps a -> good a (cld_let_bound_name_of_the_closure d)
}.

Definition env_good (env : traverse_env) (a : acc) : Prop :=
  ∀ k params x, (k, params) ∈ env_conts env -> x ∈ params -> good a x.

(* From [a] to [a'], facts are added of nodes that are not made, good in
   [a0] or made from [a0], and no read of the field of unknown arity; and
   deps of such nodes. *)
Record ext (a0 a a' : acc) : Prop := {
  ext_next : (acc_next a <= acc_next a')%positive;
  ext_facts : ∃ L, acc_facts a' = L ++ acc_facts a
    ∧ (∀ fct x, fct ∈ L -> x ∈ nodes_of fct -> ok_node a0 a' x)
    ∧ (∀ w n, Fact_accessor w unknown n ∉ L);
  ext_code_deps : ∀ id cd, (id, cd) ∈ acc_code_deps a' ->
    (id, cd) ∈ acc_code_deps a
    ∨ ∀ x, x ∈ code_dep_nodes cd -> ok_node a0 a' x;
  ext_apply_deps : ∀ d, d ∈ acc_apply_deps a' ->
    d ∈ acc_apply_deps a ∨ ∀ x, x ∈ apply_dep_nodes d -> ok_node a0 a' x;
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
Proof. left. apply AW.not_made_simple. Qed.

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
  constructor; [ lia | | | | ].
  - exists (L2 ++ L1). split; [ rewrite Hf2, Hf1, app_assoc; reflexivity | ].
    split.
    + intros fct x Hin Hx. apply in_app_or in Hin as [Hin | Hin];
        [ exact (Hi2 _ _ Hin Hx) | ].
      exact (ok_node_mono _ _ _ _ Hn2 (Hi1 _ _ Hin Hx)).
    + intros w n Hin. apply in_app_or in Hin as [Hin | Hin];
        [ exact (Hw2 _ _ Hin) | exact (Hw1 _ _ Hin) ].
  - intros id cd Hin. destruct (Hc2 _ _ Hin) as [Hin' | H]; [ | right; auto ].
    destruct (Hc1 _ _ Hin') as [H' | H']; [ left; auto | right ].
    intros x Hx. exact (ok_node_mono _ _ _ _ Hn2 (H' x Hx)).
  - intros d Hin. destruct (Hap2 _ Hin) as [Hin' | H]; [ | right; auto ].
    destruct (Hap1 _ Hin') as [H' | H']; [ left; auto | right ].
    intros x Hx. exact (ok_node_mono _ _ _ _ Hn2 (H' x Hx)).
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
  - intros id cd Hin. left. rewrite <- Hc. exact Hin.
  - intros d Hin. left. rewrite <- Hap. exact Hin.
  - intros d Hin. left. rewrite <- Hcl. exact Hin.
Qed.

Lemma ext_refl (a0 a : acc) : ext a0 a a.
Proof. apply ext_same; (reflexivity || lia). Qed.

Lemma ext_add_facts (a0 a : acc) (L : list fact) :
  (∀ fct x, fct ∈ L -> x ∈ nodes_of fct -> ok_node a0 a x) ->
  (∀ w n, Fact_accessor w unknown n ∉ L) ->
  ext a0 a (add_facts L a).
Proof.
  intros Hi Hw. constructor; cbn [acc_next acc_facts acc_code_deps
    acc_apply_deps acc_closure_deps add_facts]; [ lia | | | | ].
  - exists L. split; [ reflexivity | split; assumption ].
  - intros id cd Hin. left. exact Hin.
  - intros d Hin. left. exact Hin.
  - intros d Hin. left. exact Hin.
Qed.

Lemma ext_add_fact (a0 a : acc) (fct : fact) :
  (∀ x, x ∈ nodes_of fct -> ok_node a0 a x) ->
  (∀ w n, fct <> Fact_accessor w unknown n) ->
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
  destruct (AW.note_simples_fields l a) as (H1 & H2 & H3 & H4 & H5).
  apply ext_same; auto. rewrite H1. lia.
Qed.

Lemma ext_push_apply_dep (a0 : acc) (d : apply_dep) (a : acc) :
  (∀ x, x ∈ apply_dep_nodes d -> ok_node a0 a x) ->
  ext a0 a (push_apply_dep d a).
Proof.
  intros H. constructor; cbn; [ lia | | | | ].
  - exists []. split; [ reflexivity | split ];
      [ intros fct x [] | intros w n [] ].
  - intros id cd Hin. left. exact Hin.
  - intros d' [<- | Hin]; [ right; exact H | left; exact Hin ].
  - intros d' Hin. left. exact Hin.
Qed.

Lemma ext_push_closure_dep (a0 : acc) (d : closure_dep) (a : acc) :
  ok_node a0 a (cld_let_bound_name_of_the_closure d) ->
  ext a0 a (push_closure_dep d a).
Proof.
  intros H. constructor; cbn; [ lia | | | | ].
  - exists []. split; [ reflexivity | split ];
      [ intros fct x [] | intros w n [] ].
  - intros id cd Hin. left. exact Hin.
  - intros d' Hin. left. exact Hin.
  - intros d' [<- | Hin]; [ right; exact H | left; exact Hin ].
Qed.

Lemma ext_add_code_dep (a0 : acc) (id : code_id) (cd : code_dep)
    (a : acc) :
  (∀ x, x ∈ code_dep_nodes cd -> ok_node a0 a x) ->
  ext a0 a (add_code_dep id cd a).
Proof.
  intros H. constructor; cbn; [ lia | | | | ].
  - exists []. split; [ reflexivity | split ];
      [ intros fct x [] | intros w n [] ].
  - intros id' cd' [Heq | Hin]; [ injection Heq as <- <-; right; exact H
                                | left; exact Hin ].
  - intros d' Hin. left. exact Hin.
  - intros d' Hin. left. exact Hin.
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

(* What [ext] gives where [a0] is of the invariant. *)
Lemma good_mono (a a' : acc) (x : node) :
  (acc_next a <= acc_next a')%positive -> good a x -> good a' x.
Proof.
  intros Hle [Hlt Hs]. split; [ | exact Hs ].
  intros k Hk. specialize (Hlt k Hk). lia.
Qed.

Lemma ok_close (a0 a' : acc) (x : node) :
  uinv a0 -> (acc_next a0 <= acc_next a')%positive -> ok_node a0 a' x ->
  good a' x.
Proof.
  intros Hinv Hle [Hx | [H | (k & -> & H1 & H2)]].
  - split; [ intros k -> ; destruct (Hx k eq_refl) | ].
    intros Hs. destruct (ui_special a0 Hinv x Hs) as (k & -> & _).
    exact (Hx k eq_refl).
  - exact (good_mono _ _ _ Hle H).
  - split; [ intros k' Hk; injection Hk as <-; exact H2 | ].
    intros Hs. destruct (ui_special a0 Hinv _ Hs) as (k' & Hk & Hlt).
    injection Hk as <-. lia.
Qed.

Lemma good_avoids (a : acc) (L : list fact) :
  (∀ fct x, fct ∈ L -> x ∈ nodes_of fct -> good a x) -> avoids S L.
Proof. intros H fct y Hf Hy. exact (proj2 (H _ _ Hf Hy)). Qed.

Lemma ext_close (a0 a' : acc) : uinv a0 -> ext a0 a0 a' -> uinv a'.
Proof.
  intros Hinv He.
  assert (Hgood : ∀ x, ok_node a0 a' x -> good a' x)
    by (intros x; exact (ok_close a0 a' x Hinv (ext_next _ _ _ He))).
  destruct He as [Hn (L & Hf & Hi & Hw) Hc Hap Hcl].
  constructor.
  - intros fct x Hin Hx. rewrite Hf in Hin.
    apply in_app_or in Hin as [Hin | Hin].
    + exact (proj1 (Hgood x (Hi _ _ Hin Hx))).
    + intros k Hk. specialize (ui_nodes a0 Hinv _ _ Hin Hx k Hk). lia.
  - intros x Hs. destruct (ui_special a0 Hinv x Hs) as (k & -> & Hk).
    exists k. split; [ reflexivity | lia ].
  - intros x n Hin. rewrite Hf in Hin |- *.
    apply in_app_or in Hin as [Hin | Hin]; [ destruct (Hw _ _ Hin) | ].
    apply (chain_frame_l S S); [ auto | exact (ui_chains a0 Hinv x n Hin) | ].
    apply (good_avoids a'). intros fct y Hfct Hy.
    exact (Hgood _ (Hi _ _ Hfct Hy)).
  - intros id cd x Hin Hx. destruct (Hc _ _ Hin) as [Hin' | H].
    + exact (good_mono _ _ _ Hn (ui_code_deps a0 Hinv _ _ _ Hin' Hx)).
    + exact (Hgood _ (H x Hx)).
  - intros d x Hin Hx. destruct (Hap _ Hin) as [Hin' | H].
    + exact (good_mono _ _ _ Hn (ui_apply_deps a0 Hinv _ _ Hin' Hx)).
    + exact (Hgood _ (H x Hx)).
  - intros d Hin. destruct (Hcl _ Hin) as [Hin' | H].
    + exact (good_mono _ _ _ Hn (ui_closure_deps a0 Hinv _ Hin')).
    + exact (Hgood _ H).
Qed.

(* ------------------------------------------------------------------------ *)
(* The steps that add no read of the field of unknown arity                 *)
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
  subst; cbn [nodes_of] in Hx;
  repeat match type of Hx with
  | In _ (_ :: _) => destruct Hx as [Hx | Hx]
  | In _ [] => destruct Hx
  end;
  subst.

Ltac no_widget :=
  let w := fresh "w" in let n := fresh "n" in let H := fresh "H" in
  intros w n H; into_split H;
  repeat match goal with y : prod _ _ |- _ => destruct y end;
  try discriminate.

Ltac ok_tac :=
  first
    [ assumption
    | apply ok_simple
    | apply ok_not_made; intros ?k; discriminate
    | apply ok_made; cbn [acc_next add_fact add_facts push_apply_dep
                          push_closure_dep add_code_dep]; lia
    | eapply ok_node_mono; [ | eassumption ];
      cbn [acc_next add_fact add_facts push_apply_dep push_closure_dep
           add_code_dep]; lia ].

Ltac nodes_tac :=
  let x := fresh "x" in let Hx := fresh "Hx" in
  intros x Hx; cbn [nodes_of] in Hx;
  repeat match type of Hx with
  | In _ (_ :: _) => destruct Hx as [Hx | Hx]
  | In _ [] => destruct Hx
  end; subst; ok_tac.

Lemma ext_cond_any_usage (a0 : acc) (env : traverse_env) (s : simple)
    (a : acc) :
  ext a0 a (cond_any_usage env s a).
Proof.
  unfold cond_any_usage.
  apply (ext_trans _ _ (note_simple s a)); [ apply ext_note_simple | ].
  destruct (env_current_code_id env) as [id | ]; apply ext_add_fact;
    try (intros ? ? ?; discriminate); nodes_tac.
Qed.

Lemma ext_cond_alias (a0 : acc) (env : traverse_env) (from to_ : node)
    (a : acc) :
  ok_node a0 a from -> ok_node a0 a to_ -> ext a0 a (cond_alias env from to_ a).
Proof.
  intros H1 H2. unfold cond_alias.
  destruct (env_current_code_id env) as [id | ]; apply ext_add_fact;
    try (intros ? ? ?; discriminate); nodes_tac.
Qed.

Lemma ext_fresh_list (a0 : acc) (n : nat) (a : acc) :
  (acc_next a0 <= acc_next a)%positive ->
  ext a0 a (snd (fresh_list n a))
  ∧ ∀ x, x ∈ fst (fresh_list n a) -> ok_node a0 (snd (fresh_list n a)) x.
Proof.
  intros Hle.
  destruct (AW.fresh_list_spec n a) as (H1 & H2 & H3 & H4 & H5 & H6).
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
    exact (Henv k params x (AW.lookup_in_conts _ _ _ E) Hx).
  - destruct (ext_fresh_list a0 count a Hle) as [H1 H2].
    destruct (fresh_list count a) as [params b] eqn:Eb.
    cbn [fst snd] in *. split.
    + apply (ext_trans _ _ _ _ H1). apply ext_add_facts.
      * intros fct x Hin Hx. into_split Hin. into_close Hin Hx.
        apply H2. assumption.
      * no_widget.
    + intros x Hx. exact (H2 x Hx).
Qed.

Lemma ext_block_deps (a0 : acc) (base : node) (fields : list simple)
    (a : acc) :
  ok_node a0 a base -> ext a0 a (block_deps base fields a).
Proof.
  intros H. unfold block_deps.
  destruct (AW.note_simples_fields fields a) as (H1 & _).
  apply (ext_trans _ _ (note_simples fields a));
    [ apply ext_note_simples | ].
  apply ext_add_facts.
  - intros fct x Hin Hx. into_split Hin; into_close Hin Hx;
      try ok_tac; eapply ok_node_mono; try eassumption; rewrite H1; lia.
  - no_widget.
Qed.

Lemma free_names_not_made (args : list simple) (y : node) :
  y ∈ free_names_of_simples args -> not_made y.
Proof.
  unfold free_names_of_simples. intros H. apply in_flat_map in H.
  destruct H as [s [_ H]]. destruct s; cbn in H;
    try contradiction; destruct H as [<- | []]; intros k; discriminate.
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
    - nodes_tac.
    - intros w n Heq. injection Heq as _ Hf' _. exact (Hf Hf'). }
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
      + intros fct y Hin Hy. into_split Hin; into_close Hin Hy; try ok_tac.
        apply ok_not_made. eapply free_names_not_made. eassumption.
      + no_widget. }
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
    apply ext_add_fact; [ nodes_tac | ].
    intros ? ? ?; discriminate.
  - apply (ext_trans _ _ (note_simple s a)); [ apply ext_note_simple | ].
    apply ext_add_fact; [ nodes_tac | ].
    intros ? ? ?; discriminate.
Qed.


Lemma ext_record_set (a0 : acc) (env : traverse_env)
    (closures : list (node * option code_id))
    (function_slots : list (function_slot * node))
    (value_slots : list (value_slot * simple)) (a : acc) :
  (∀ base code, (base, code) ∈ closures -> ok_node a0 a base) ->
  (∀ f c, (f, c) ∈ function_slots -> ok_node a0 a c) ->
  ext a0 a (record_set_of_closures_deps env closures function_slots
              value_slots a).
Proof.
  intros H Hfs. unfold record_set_of_closures_deps. apply ext_fold.
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
    destruct (AW.note_simples_fields l b1) as (Hn & _);
    apply (ext_trans _ _ (note_simples l b1)); [ apply ext_note_simples | ]
  end.
  pose proof (ext_next _ _ _ H1) as Hn1.
  apply ext_add_facts.
  - intros fct x Hfct Hx. into_split Hfct; into_close Hfct Hx; try ok_tac;
      apply (ok_node_mono _ b); try (rewrite Hn; exact Hn1); try exact Hb.
    apply (ok_node_mono _ a); [ exact Hle | eapply Hfs; eassumption ].
  - no_widget.
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
  apply ext_record_set.
  - intros base code Hin.
    apply in_map_iff in Hin. destruct Hin as [[[f name] id] [Heq Hin]].
    injection Heq as <- _. exact (H2 _ _ _ Hin).
  - intros f c Hin.
    apply in_map_iff in Hin. destruct Hin as [[[f' name] id] [Heq Hin]].
    injection Heq as <- <-. exact (H2 _ _ _ Hin).
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
    apply ext_add_fact; [ nodes_tac | ].
    intros ? ? ?; discriminate.
  - apply ext_traverse_prim.
  - apply ext_record_set; [ | intros f c [] ]. intros base code [Heq | []].
    injection Heq as <- _. ok_tac.
  - apply ext_traverse_closure_of_set. exact Hle.
  - apply (ext_trans _ _ (note_simple c a)); [ apply ext_note_simple | ].
    apply ext_add_fact; [ nodes_tac | ].
    intros ? ? ?; discriminate.
  - apply (ext_trans _ _ (note_simple c a)); [ apply ext_note_simple | ].
    apply ext_add_fact; [ nodes_tac | ].
    intros ? ? ?; discriminate.
  - apply (ext_trans _ _ (note_simple c a)); [ apply ext_note_simple | ].
    apply ext_add_fact; [ nodes_tac | ].
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
  - apply ext_record_set; [ | intros f c [] ]. intros base code [Heq | []].
    injection Heq as <- _. ok_tac.
  - apply ext_record_set.
    + intros base code [Heq | []]. injection Heq as <- _. ok_tac.
    + intros f c Hin. apply in_map_iff in Hin.
      destruct Hin as [[f' sym] [Heq _]]. injection Heq as _ <-. ok_tac.
  - apply ext_block_deps. ok_tac.
  - unfold boxed_number_facts. apply ext_add_facts.
    + intros fct y Hin Hy. repeat match goal with
      | H : context [match ?t with _ => _ end] |- _ => destruct t
      end; into_split Hin; into_close Hin Hy; ok_tac.
    + intros w n H. repeat match goal with
      | H : context [match ?t with _ => _ end] |- _ => destruct t
      end; into_split H; discriminate.
  - apply ext_add_fact; [ nodes_tac | intros ? ? ?; discriminate ].
  - apply ext_add_fact; [ nodes_tac | intros ? ? ?; discriminate ].
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
  (∀ x, x ∈ exn :: params ++ returns -> ok_node a0 a x) ->
  ext a0 a (snd (create_known_arity_call_witness id params returns exn a))
  ∧ ok_node a0 (snd (create_known_arity_call_witness id params returns exn a))
      (fst (create_known_arity_call_witness id params returns exn a)).
Proof.
  intros Hle Hp. unfold create_known_arity_call_witness.
  fresh_step a0 a w b E Hw. cbv zeta. cbn [fst snd].
  split; [ | exact Ho ].
  apply (ext_trans _ _ _ _ He). apply ext_add_facts; [ | no_widget ].
  intros fct x Hin Hx. into_split Hin; into_close Hin Hx;
    try exact Ho; try ok_tac;
    (eapply ok_node_mono; [ | apply Hp ]; [ rewrite Hn; lia | ]);
    try (left; reflexivity);
    right; apply in_or_app;
    first [ left; eapply AW.indexed_in; eassumption
          | right; eapply AW.indexed_in; eassumption ].
Qed.

Lemma ext_create_tupled (a0 : acc) (id : code_id) (params returns : list node)
    (exn : node) (a : acc) :
  (acc_next a0 <= acc_next a)%positive ->
  (∀ x, x ∈ exn :: params ++ returns -> ok_node a0 a x) ->
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
  assert (Hpd : ∀ x, x ∈ exn :: params ++ returns -> ok_node a0 d x).
  { intros x Hx. eapply ok_node_mono; [ | exact (Hp x Hx) ]. lia. }
  assert (Hwd : ok_node a0 d w) by (eapply ok_node_mono; [ | exact Ho ]; lia).
  assert (Hud : ok_node a0 d u)
    by (eapply ok_node_mono; [ | exact Ho0 ]; lia).
  split.
  - apply (ext_trans _ _ _ _ He). apply (ext_trans _ _ _ _ He0).
    apply (ext_trans _ _ _ _ He1). apply ext_add_facts; [ | no_widget ].
    intros fct x Hin Hx. into_split Hin; into_close Hin Hx;
      try ok_tac; apply Hpd;
      try (left; reflexivity);
      right; apply in_or_app;
      first [ left; eapply AW.indexed_in; eassumption
            | right; eapply AW.indexed_in; eassumption ].
  - intros x [<- | []]. cbn [acc_next add_facts].
    exact Hwd.
Qed.

Lemma ext_witness_deps (a0 : acc) (id : code_id) (returns : list node)
    (exn : node) (pw : list (list node * node)) (a : acc) :
  (acc_next a0 <= acc_next a)%positive ->
  (∀ x, x ∈ exn :: returns -> ok_node a0 a x) ->
  (∀ first w, (first, w) ∈ pw ->
     ok_node a0 a w ∧ ∀ x, x ∈ first -> ok_node a0 a x) ->
  ext a0 a (unknown_arity_witness_deps id returns exn pw a).
Proof.
  revert a. induction pw as [ | [first w] rest IH]; intros a Hle Her Hpw;
    cbn [unknown_arity_witness_deps]; [ apply ext_refl | ].
  destruct (Hpw first w (or_introl eq_refl)) as [Hw Hfirst].
  set (b := add_facts _ a).
  assert (Hbn : acc_next b = acc_next a) by reflexivity.
  assert (H1 : ext a0 a b).
  { apply ext_add_facts; [ | no_widget ].
    intros fct x Hin Hx. into_split Hin; into_close Hin Hx; try exact Hw;
      try ok_tac.
    apply Hfirst. eapply AW.indexed_in. eassumption. }
  apply (ext_trans _ _ _ _ H1).
  assert (Hrest : ∀ first' w', (first', w') ∈ rest ->
            ok_node a0 b w' ∧ ∀ x, x ∈ first' -> ok_node a0 b x).
  { intros first' w' Hin. destruct (Hpw _ _ (or_intror Hin)) as [H2 H3].
    split; [ exact H2 | exact H3 ]. }
  destruct rest as [ | [g nw] rest'].
  - apply ext_add_facts; [ | no_widget ].
    intros fct x Hin Hx. into_split Hin; into_close Hin Hx; try exact Hw;
      apply Her; [ left; reflexivity | right; eapply AW.indexed_in;
                                        eassumption ].
  - fresh_step a0 b v c E Hv.
    cbv zeta.
    destruct (Hrest g nw (or_introl eq_refl)) as [Hnw _].
    assert (Hcv : ok_node a0 c v) by exact Ho.
    set (d := add_facts _ c).
    assert (Hdn : acc_next d = acc_next c) by reflexivity.
    assert (H2 : ext a0 c d).
    { apply ext_add_facts; [ | no_widget ].
      intros fct x Hin Hx. into_split Hin; into_close Hin Hx; try exact Hcv;
        eapply ok_node_mono; try eassumption; lia. }
    apply (ext_trans _ _ _ _ He). apply (ext_trans _ _ _ _ H2).
    apply IH; [ lia | | ].
    + intros x Hx. eapply ok_node_mono; [ | exact (Her x Hx) ]. lia.
    + intros first' w' Hin. destruct (Hrest _ _ Hin) as [H3 H4].
      split; [ eapply ok_node_mono; [ | exact H3 ] | intros x Hx;
               eapply ok_node_mono; [ | exact (H4 x Hx) ] ]; lia.
Qed.

Lemma ext_create_non_tupled (a0 : acc) (id : code_id) (ar : arity)
    (params returns : list node) (exn : node) (a : acc) :
  (acc_next a0 <= acc_next a)%positive ->
  (∀ x, x ∈ exn :: params ++ returns -> ok_node a0 a x) ->
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
  { apply ext_witness_deps; [ lia | | ].
    - intros x Hx. eapply ok_node_mono; [ exact Hn1 | ]. apply Hp.
      destruct Hx as [<- | Hx]; [ left; reflexivity | ].
      right. apply in_or_app. right. exact Hx.
    - intros first w Hin. split.
      + exact (H1o w (in_combine_r _ _ _ _ Hin)).
      + intros x Hx. apply in_combine_l in Hin.
        eapply ok_node_mono; [ exact Hn1 | ].
        apply Hp. right. apply in_or_app. left.
        exact (AW.group_in ar params first x Hin Hx). }
  split; [ exact (ext_trans _ _ _ _ H1 H2) | ].
  intros x Hx. eapply ok_node_mono; [ exact (ext_next _ _ _ H2) | ].
  exact (H1o x Hx).
Qed.

Lemma ext_create_unknown (a0 : acc) (id : code_id) (is_tupled : bool)
    (ar : arity) (params returns : list node) (exn : node) (a : acc) :
  (acc_next a0 <= acc_next a)%positive ->
  (∀ x, x ∈ exn :: params ++ returns -> ok_node a0 a x) ->
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

Lemma ext_prepare_code (a0 : acc) (id : code_id) (c : code) (a : acc) :
  (acc_next a0 <= acc_next a)%positive ->
  ext a0 a (prepare_code id c a).
Proof.
  intros Hle. unfold prepare_code.
  destruct (ext_fresh_list a0 (length (code_result_kinds c)) a Hle)
    as [H1 H1o].
  destruct (fresh_list _ a) as [returns b1] eqn:E1. cbn [fst snd] in *.
  cbv beta iota.
  fresh_step a0 b1 exn b2 E2 Hexn. cbv beta iota.
  fresh_step a0 b2 mc b3 E3 Hmc. cbv beta iota.
  destruct (ext_fresh_list a0 (length (code_params_kinds c)) b3)
    as [H4 H4o]; [ nexts; lia | ].
  destruct (fresh_list _ b3) as [params b4] eqn:E4. cbn [fst snd] in *.
  cbv beta iota.
  assert (Hall : ∀ x, x ∈ exn :: params ++ returns -> ok_node a0 b4 x).
  { intros x [<- | Hx]; [ mono | ].
    apply in_app_or in Hx as [Hx | Hx]; [ exact (H4o x Hx) | ].
    pose proof (H1o x Hx). mono. }
  destruct (ext_create_known a0 id params returns exn b4) as [H5 H5o];
    [ nexts; lia | exact Hall | ].
  destruct (create_known_arity_call_witness id params returns exn b4)
    as [known b5] eqn:E5. cbn [fst snd] in *. cbv beta iota.
  destruct (ext_create_unknown a0 id (code_is_tupled c) (code_params_arity c)
              params returns exn b5) as [H6 H6o];
    [ nexts; lia | intros x Hx; pose proof (Hall x Hx); mono | ].
  destruct (create_unknown_arity_call_witnesses _ _ _ _ _ _ b5)
    as [unknown b6] eqn:E6. cbn [fst snd] in *. cbv beta iota zeta.
  apply (ext_trans _ _ _ _ H1). apply (ext_trans _ _ _ _ He).
  apply (ext_trans _ _ _ _ He0). apply (ext_trans _ _ _ _ H4).
  apply (ext_trans _ _ _ _ H5). apply (ext_trans _ _ _ _ H6).
  match goal with
  | |- ext _ _ (add_code_dep _ _ ?b) => apply (ext_trans _ _ b)
  end.
  { apply ext_add_fact; [ nodes_tac | intros ? ? ?; discriminate ]. }
  apply ext_add_code_dep.
  cbn [code_dep_nodes cd_my_closure cd_exn cd_return cd_params
       cd_known_arity_call_witness cd_unknown_arity_call_witnesses acc_next
       add_fact add_facts].
  intros x [<- | [<- | [<- | Hx]]]; [ mono | mono | mono | ].
  apply in_app_or in Hx as [Hx | Hx].
  - pose proof (H4o x Hx). mono.
  - apply in_app_or in Hx as [Hx | Hx].
    + pose proof (H1o x Hx). mono.
    + pose proof (H6o x Hx). mono.
Qed.

Lemma ext_prepare_binding (a0 : acc) (binding : static_binding) (a : acc) :
  (acc_next a0 <= acc_next a)%positive ->
  ext a0 a (prepare_binding binding a).
Proof.
  intros Hle. destruct binding; cbn [prepare_binding];
    [ apply ext_prepare_code; exact Hle | apply ext_refl | apply ext_refl ].
Qed.


(* ------------------------------------------------------------------------ *)
(* Calls                                                                    *)
(* ------------------------------------------------------------------------ *)

Lemma ext_make_known (a0 : acc) (env : traverse_env) (ap : apply_expr)
    (returns : list node) (exn : node) (a : acc) :
  (acc_next a0 <= acc_next a)%positive ->
  (∀ x, x ∈ exn :: returns -> ok_node a0 a x) ->
  ext a0 a (snd (make_known_arity_apply_widget env ap returns exn a))
  ∧ ok_node a0 (snd (make_known_arity_apply_widget env ap returns exn a))
      (fst (make_known_arity_apply_widget env ap returns exn a)).
Proof.
  intros Hle Hin. unfold make_known_arity_apply_widget.
  fresh_step a0 a wit b1 E1 Hw. cbv beta iota.
  fresh_step a0 b1 called b2 E2 Hc. cbv beta iota.
  fresh_step a0 b2 an b3 E3 Han. cbv beta iota zeta.
  pose proof (ext_note_simples a0 (ap_args ap) b3) as H4.
  destruct (AW.note_simples_fields (ap_args ap) b3) as (N1 & _).
  set (b4 := note_simples (ap_args ap) b3) in *. clearbody b4.
  set (b5 := add_facts _ b4).
  assert (H5 : ext a0 b4 b5).
  { apply ext_add_facts; [ | no_widget ].
    intros fct x H Hx. into_split H; into_close H Hx; try ok_tac;
      eapply ok_node_mono; try (apply Hin; first [ left; reflexivity
        | right; eapply AW.indexed_in; eassumption ]); lia. }
  assert (H6 : ext a0 b5 (cond_alias env an wit b5)).
  { apply ext_cond_alias; unfold b5; cbn [add_facts acc_next]; ok_tac. }
  cbn [fst snd]. split.
  - apply (ext_trans _ _ _ _ He). apply (ext_trans _ _ _ _ He0).
    apply (ext_trans _ _ _ _ He1). apply (ext_trans _ _ _ _ H4).
    apply (ext_trans _ _ _ _ H5). exact H6.
  - apply (ok_node_mono _ b5);
      [ | unfold b5; cbn [add_facts acc_next]; ok_tac ].
    unfold cond_alias. destruct (env_current_code_id env); cbn; lia.
Qed.

Lemma ext_add_apply (a0 : acc) (env : traverse_env) (id : code_id)
    (callee call_widget : node) (b : bool) (a : acc) :
  (acc_next a0 <= acc_next a)%positive ->
  ok_node a0 a callee -> ok_node a0 a call_widget ->
  ext a0 a (add_apply env id callee call_widget b a).
Proof.
  intros Hle H1 H2. unfold add_apply. destruct b.
  - fresh_step a0 a cif b1 E1 Hcif. cbv beta iota.
    fresh_step a0 b1 wif b2 E2 Hwif. cbv beta iota.
    apply (ext_trans _ _ _ _ He). apply (ext_trans _ _ _ _ He0).
    set (b3 := add_facts _ b2).
    assert (H3 : ext a0 b2 b3).
    { apply ext_add_facts; [ | no_widget ].
      intros fct x H Hx. into_split H; into_close H Hx; ok_tac. }
    apply (ext_trans _ _ _ _ H3). apply ext_push_apply_dep.
    unfold b3. cbn [apply_dep_nodes ad_apply_closure ad_apply_call_witness
      acc_next add_facts].
    intros x [<- | [<- | []]]; ok_tac.
  - apply ext_push_apply_dep. cbn [apply_dep_nodes ad_apply_closure
      ad_apply_call_witness]. intros x [<- | [<- | []]]; ok_tac.
Qed.

(* The calls of known arity add no read of the field of unknown arity. *)
Lemma ext_call_known (a0 : acc) (env : traverse_env) (ap : apply_expr)
    (exn_arg : node) (return_args : list node) (a : acc) :
  ap_call_kind ap <> Indirect_unknown_arity ->
  (acc_next a0 <= acc_next a)%positive ->
  (∀ x, x ∈ exn_arg :: return_args -> ok_node a0 a x) ->
  ext a0 a (traverse_call_kind env ap exn_arg return_args a).
Proof.
  intros Hkind Hle Hin. unfold traverse_call_kind. cbv zeta.
  pose proof (ext_note_simple a0 (ap_callee ap) a) as H1.
  destruct (AW.note_simple_fields (ap_callee ap) a) as (N1 & _).
  set (a1 := note_simple (ap_callee ap) a) in *. clearbody a1.
  assert (Hin1 : ∀ x, x ∈ exn_arg :: return_args -> ok_node a0 a1 x).
  { intros x Hx. pose proof (Hin x Hx). mono. }
  destruct (ap_call_kind ap) as [id | | code_ids];
    [ | destruct (Hkind eq_refl) | ].
  - destruct (ext_make_known a0 env ap return_args exn_arg a1)
      as [Hk Hcw]; [ lia | exact Hin1 | ].
    destruct (make_known_arity_apply_widget env ap return_args exn_arg a1)
      as [cw a2] eqn:E. cbn [fst snd] in *. cbv beta iota zeta.
    set (a3 := add_fact _ a2).
    assert (H3 : ext a0 a2 a3).
    { apply ext_add_fact; [ nodes_tac | intros ? ? ?; discriminate ]. }
    apply (ext_trans _ _ _ _ H1). apply (ext_trans _ _ _ _ Hk).
    apply (ext_trans _ _ _ _ H3).
    assert (Hn3 : (acc_next a0 <= acc_next a3)%positive)
      by (nexts; unfold a3; cbn [add_fact add_facts acc_next]; lia).
    assert (Hc3 : ok_node a0 a3 (node_of_simple (ap_callee ap)))
      by ok_tac.
    assert (Hw3 : ok_node a0 a3 cw)
      by (unfold a3; cbn [add_fact add_facts]; ok_tac).
    destruct (env_should_preserve_direct_calls env);
      try (apply ext_add_apply; assumption).
    destruct (negb _); [ apply ext_add_apply; assumption | apply ext_refl ].
  - destruct (ext_make_known a0 env ap return_args exn_arg a1)
      as [Hk Hcw]; [ lia | exact Hin1 | ].
    destruct (make_known_arity_apply_widget env ap return_args exn_arg a1)
      as [cw a2] eqn:E. cbn [fst snd] in *. cbv beta iota zeta.
    apply (ext_trans _ _ _ _ H1). apply (ext_trans _ _ _ _ Hk).
    apply ext_add_fact; [ nodes_tac | intros ? ? ?; discriminate ].
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
  apply ext_add_facts; [ | no_widget ].
  intros fct y H Hy. into_split H; into_close H Hy; try ok_tac.
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
  apply ext_add_fact; [ nodes_tac | ].
  intros ? ? ?; discriminate.
Qed.

Lemma ext_resolve (a : acc) (unit_symbols : list symbol) :
  uinv a -> ext a a (resolve_delayed_deps unit_symbols a).
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
      assert (Hw : ∀ x, x ∈ apply_dep_nodes d -> ok_node a a x)
        by (intros x Hx;
            exact (ok_good _ _ _ (ui_apply_deps a Hinv d x Hd Hx))).
      cbn [apply_dep_nodes] in Hw.
      pose proof (Hw _ (or_introl eq_refl)) as Hw1.
      pose proof (Hw _ (or_intror (or_introl eq_refl))) as Hw2.
      unfold apply_dep_facts, alias_for_caller in H.
      destruct (lookup (ad_apply_code_id d) (acc_code_deps a)) as [cd | ]
        eqn:E;
        destruct (ad_function_containing_apply_expr d);
        into_split H; into_close H Hy; try exact Hw1; try exact Hw2;
        try ok_tac; apply (Hcd _ cd _ E);
        first [ apply in_cd_my | apply in_cd_known ].
    - intros w n H. apply in_flat_map in H. destruct H as [d [_ H]].
      unfold apply_dep_facts, alias_for_caller in H.
      destruct (lookup _ _); destruct (ad_function_containing_apply_expr d);
        into_split H; discriminate. }
  assert (Hcl : ∀ d, d ∈ acc_closure_deps b1 ->
                  ok_node a a (cld_let_bound_name_of_the_closure d)).
  { intros d Hd. apply ok_good. exact (ui_closure_deps a Hinv d Hd). }
  set (b2 := List.fold_left _ (acc_closure_deps b1) b1).
  assert (H2 : ext a b1 b2).
  { apply ext_fold. intros d b Hd Hb. unfold add_closure_dep.
    pose proof (Hcl d Hd) as Hc.
    destruct (lookup (cld_closure_code_id d) (acc_code_deps a)) as [cd | ]
      eqn:E.
    - apply ext_add_facts.
      + intros fct y H Hy.
        assert (Hb' : (acc_next a <= acc_next b)%positive)
          by (unfold b1 in Hb; cbn [acc_next add_facts] in Hb; lia).
        assert (Huw : ∀ uw uws, cd_unknown_arity_call_witnesses cd = uw :: uws
                  -> ok_node a b uw).
        { intros uw uws Heq. eapply ok_node_mono; [ exact Hb' | ].
          apply (Hcd _ cd _ E). apply in_cd_unknown. rewrite Heq.
          left. reflexivity. }
        assert (Hmy : ok_node a b (cd_my_closure cd))
          by (eapply ok_node_mono; [ exact Hb' | ];
              apply (Hcd _ cd _ E); apply in_cd_my).
        assert (Hkn : ok_node a b (cd_known_arity_call_witness cd))
          by (eapply ok_node_mono; [ exact Hb' | ];
              apply (Hcd _ cd _ E); apply in_cd_known).
        assert (Hc' : ok_node a b (cld_let_bound_name_of_the_closure d))
          by (eapply ok_node_mono; [ exact Hb' | exact Hc ]).
        destruct (cd_unknown_arity_call_witnesses cd) as [ | uw uws] eqn:Euw;
          into_split H; into_close H Hy; try ok_tac.
        exact (Huw _ _ eq_refl).
      + intros w n H. into_split H; discriminate.
    - fresh_step a b w b' E' Hw'.
      cbv beta iota. apply (ext_trans _ _ _ _ He).
      apply ext_add_facts; [ | no_widget ].
      intros fct y H Hy. into_split H; into_close H Hy; try exact Ho;
        try ok_tac.
      all: eapply ok_node_mono; [ | exact Hc ];
        unfold b1 in Hb; cbn [acc_next add_facts] in Hb; lia. }
  apply (ext_trans _ _ _ _ H1). apply (ext_trans _ _ _ _ H2).
  apply ext_add_facts; [ | no_widget ].
  intros fct y H Hy. into_split H; into_close H Hy; ok_tac.
Qed.

End Ext.

(* ------------------------------------------------------------------------ *)
(* The chain of a call of unknown arity                                     *)
(* ------------------------------------------------------------------------ *)

(* The nodes made from [p] to [q]. *)
Definition fresh_in (p q : positive) (x : node) : Prop :=
  ∃ k, x = Node_made k ∧ (p <= k)%positive ∧ (k < q)%positive.

Lemma fresh_in_lt (p q : positive) (x : node) :
  fresh_in p q x -> made_lt x p -> False.
Proof. intros (k & -> & H1 & H2) H. specialize (H k eq_refl). lia. Qed.

Lemma fresh_in_mono (p p' q q' : positive) (x : node) :
  (p' <= p)%positive -> (q <= q')%positive -> fresh_in p q x ->
  fresh_in p' q' x.
Proof.
  intros H1 H2 (k & -> & H3 & H4). exists k. split; [ reflexivity | lia ].
Qed.

Lemma uniq_into_app (L1 L2 : list fact) (x : node) (fct : fact) :
  uniq_into L1 x fct -> (∀ f, f ∈ L2 -> x ∉ into_nodes f) ->
  uniq_into (L1 ++ L2) x fct.
Proof.
  intros [H1 H2] H3. split; [ apply in_or_app; left; exact H1 | ].
  intros f Hf Hx. apply in_app_or in Hf as [Hf | Hf]; [ exact (H2 _ Hf Hx) | ].
  destruct (H3 _ Hf Hx).
Qed.

Lemma uniq_into_app_r (L1 L2 : list fact) (x : node) (fct : fact) :
  uniq_into L2 x fct -> (∀ f, f ∈ L1 -> x ∉ into_nodes f) ->
  uniq_into (L1 ++ L2) x fct.
Proof.
  intros [H1 H2] H3. split; [ apply in_or_app; right; exact H1 | ].
  intros f Hf Hx. apply in_app_or in Hf as [Hf | Hf]; [ | exact (H2 _ Hf Hx) ].
  destruct (H3 _ Hf Hx).
Qed.

(* The witnesses are distinct and made before [p], and the nodes of [E]
   are made before [p] and no witness. *)
Definition sep_ok (E ws : list node) (p : positive) : Prop :=
  NoDup ws ∧ (∀ w, w ∈ ws -> made_lt w p)
  ∧ ∀ x, x ∈ E -> made_lt x p ∧ x ∉ ws.

Ltac fsplit H :=
  repeat match type of H with
  | In _ (_ ++ _) => apply in_app_or in H; destruct H as [H | H]
  | In _ (map _ _) =>
    let y := fresh "y" in let Hy := fresh "Hy" in
    apply in_map_iff in H; destruct H as [y [H Hy]]
  | In _ (_ :: _) => destruct H as [H | H]
  | In _ [] => destruct H
  end;
  repeat match goal with y : prod _ _ |- _ => destruct y end;
  subst.

Lemma deps_spec (returns : list node) (exn called : node)
    (aw : list (list simple * node)) (a : acc) :
  sep_ok (exn :: called :: returns) (map snd aw) (acc_next a) ->
  ∃ D,
    acc_facts (unknown_arity_apply_deps returns exn called aw a)
    = D ++ acc_facts a
    ∧ acc_code_deps (unknown_arity_apply_deps returns exn called aw a)
      = acc_code_deps a
    ∧ acc_apply_deps (unknown_arity_apply_deps returns exn called aw a)
      = acc_apply_deps a
    ∧ acc_closure_deps (unknown_arity_apply_deps returns exn called aw a)
      = acc_closure_deps a
    ∧ (acc_next a
       <= acc_next (unknown_arity_apply_deps returns exn called aw a))%positive
    ∧ (∀ x n, Fact_accessor x unknown n ∈ D ->
         uniq_into D x (Fact_accessor x unknown n)
         ∧ (∃ w, uniq_into D n (Fact_accessor n first_result w))
         ∧ x ∈ tl (map snd aw)
         ∧ fresh_in (acc_next a)
             (acc_next (unknown_arity_apply_deps returns exn called aw a)) n)
    ∧ (∀ fct x, fct ∈ D -> x ∈ into_nodes fct ->
         x ∈ exn :: called :: returns
         ∨ fresh_in (acc_next a)
             (acc_next (unknown_arity_apply_deps returns exn called aw a)) x
         ∨ x ∈ tl (map snd aw))
    ∧ (∀ w y, ~ aliased_from D w y)
    ∧ (∀ c y, Fact_accessor c cid y ∈ D -> y ∈ map snd aw)
    ∧ (∀ fct x, fct ∈ D -> x ∈ nodes_of fct ->
         x ∈ exn :: called :: returns
         ∨ fresh_in (acc_next a)
             (acc_next (unknown_arity_apply_deps returns exn called aw a)) x
         ∨ x ∈ map snd aw ∨ not_made x).
Proof.
  revert a. induction aw as [ | [first w] rest IH]; intros a Hsep;
    cbn [unknown_arity_apply_deps].
  - exists []. cbn [app]. repeat split; auto; try lia;
      try (intros; contradiction).
    intros w y [H | [c H]]; destruct H.
  - destruct Hsep as (Hnd & Hws & HE).
    cbn [map snd] in Hnd, Hws, HE |- *.
    apply NoDup_cons_iff in Hnd as [Hwn Hnd].
    set (F1 := map (fun '(i, s) => Fact_argument (node_of_simple s) i w)
                 (indexed first)
               ++ [Fact_accessor exn (Field_return_of_call Exn) w;
                   Fact_accessor called cid w]).
    assert (HF1i : ∀ f x, f ∈ F1 -> x ∈ into_nodes f ->
                     x = exn ∨ x = called).
    { intros f x Hf Hx. unfold F1 in Hf. fsplit Hf; cbn in Hx;
        repeat destruct Hx as [<- | Hx]; auto; contradiction. }
    assert (HF1n : ∀ f x, f ∈ F1 -> x ∈ nodes_of f ->
                     x = exn ∨ x = called ∨ x = w ∨ not_made x).
    { intros f x Hf Hx. unfold F1 in Hf. fsplit Hf; cbn in Hx;
        repeat destruct Hx as [<- | Hx]; auto; try contradiction.
      right; right; right. apply AW.not_made_simple. }
    assert (HF1u : ∀ x n, Fact_accessor x unknown n ∉ F1).
    { intros x n Hf. unfold F1 in Hf. fsplit Hf; discriminate. }
    assert (HF1a : ∀ v y, ~ aliased_from F1 v y).
    { intros v y [Hf | [c Hf]]; unfold F1 in Hf; fsplit Hf; discriminate. }
    assert (HF1c : ∀ c y, Fact_accessor c cid y ∈ F1 -> y = w).
    { intros c y Hf. unfold F1 in Hf. fsplit Hf; try discriminate.
      injection Hf as _ <-. reflexivity. }
    assert (HEr : ∀ x, x ∈ exn :: called :: returns -> x ∉ map snd rest).
    { intros x Hx Hr. apply (proj2 (HE x Hx)). right. exact Hr. }
    destruct rest as [ | [g nw] rest'].
    + set (F2 := map (fun '(i, r) =>
                        Fact_accessor r (Field_return_of_call (Normal i)) w)
                   (indexed returns)).
      exists (F2 ++ F1). cbn [add_facts acc_facts acc_code_deps
        acc_apply_deps acc_closure_deps acc_next].
      split; [ rewrite app_assoc; reflexivity | ].
      split; [ reflexivity | ]. split; [ reflexivity | ].
      split; [ reflexivity | ]. split; [ lia | ].
      split; [ | split; [ | split; [ | split ] ] ].
      * intros x n Hf. exfalso. apply in_app_or in Hf as [Hf | Hf];
          [ unfold F2 in Hf; fsplit Hf; discriminate | exact (HF1u _ _ Hf) ].
      * intros f x Hf Hx. left.
        apply in_app_or in Hf as [Hf | Hf].
        -- unfold F2 in Hf. fsplit Hf. cbn in Hx.
           destruct Hx as [<- | []]. right; right.
           eapply AW.indexed_in. eassumption.
        -- destruct (HF1i _ _ Hf Hx) as [-> | ->]; [ left | right; left ];
             reflexivity.
      * intros v y [Hf | [c Hf]]; apply in_app_or in Hf as [Hf | Hf];
          try (unfold F2 in Hf; fsplit Hf; discriminate);
          apply (HF1a v y); [ left | right; exists c ]; exact Hf.
      * intros c y Hf. apply in_app_or in Hf as [Hf | Hf];
          [ unfold F2 in Hf; fsplit Hf; discriminate | ].
        left. symmetry. exact (HF1c _ _ Hf).
      * intros f x Hf Hx. apply in_app_or in Hf as [Hf | Hf].
        -- unfold F2 in Hf. fsplit Hf. cbn in Hx.
           destruct Hx as [<- | [<- | []]].
           ++ left. right; right. eapply AW.indexed_in. eassumption.
           ++ right; right; left. left. reflexivity.
        -- destruct (HF1n _ _ Hf Hx) as [-> | [-> | [-> | H]]].
           ++ left. left. reflexivity.
           ++ left. right; left. reflexivity.
           ++ right; right; left. left. reflexivity.
           ++ right; right; right. exact H.
    + cbn [fresh]. cbv beta iota zeta.
      set (v := Node_made (acc_next a)).
      set (F3 := [Fact_accessor v first_result w;
                  Fact_accessor nw unknown v]).
      set (c := {| acc_next := Pos.succ (acc_next a);
                   acc_facts := F3 ++ F1 ++ acc_facts a;
                   acc_code_deps := acc_code_deps a;
                   acc_apply_deps := acc_apply_deps a;
                   acc_closure_deps := acc_closure_deps a;
                   acc_symbols := acc_symbols a |}).
      cbn [map snd] in Hnd, Hws, HE, Hwn, HEr.
      set (a'' := unknown_arity_apply_deps returns exn called
                    ((g, nw) :: rest') _).
      assert (Ea : a'' = unknown_arity_apply_deps returns exn called
                           ((g, nw) :: rest') c) by reflexivity.
      clearbody a''. subst a''.
      apply NoDup_cons_iff in Hnd as [Hnwn Hnd'].
      assert (Hlt : ∀ x, made_lt x (acc_next a) ->
                      made_lt x (acc_next c)).
      { intros x Hx. apply (AW.made_lt_mono _ _ _ Hx). cbn. lia. }
      assert (Hsep' : sep_ok (exn :: called :: returns)
                        (map snd ((g, nw) :: rest')) (acc_next c)).
      { split; [ cbn [map]; apply NoDup_cons; assumption | ].
        split.
        - intros x Hx. apply Hlt. apply Hws. right. exact Hx.
        - intros x Hx. split; [ apply Hlt; exact (proj1 (HE x Hx)) | ].
          exact (HEr x Hx). }
      destruct (IH c Hsep') as (D' & HD & Hcd & Had & Hcl & Hn & HD1 & HD2
                                & HD3 & HD4 & HD5).
      set (a' := unknown_arity_apply_deps returns exn called
                   ((g, nw) :: rest') c) in *.
      assert (Hca : acc_next c = Pos.succ (acc_next a)) by reflexivity.
      assert (Hwid : ∀ x, fresh_in (acc_next c) (acc_next a') x ->
                       fresh_in (acc_next a) (acc_next a') x).
      { intros x Hx.
        apply (fresh_in_mono (acc_next c) _ (acc_next a'));
          [ | lia | exact Hx ].
        rewrite Hca. lia. }
      assert (Hvf : fresh_in (acc_next a) (acc_next a') v).
      { exists (acc_next a). split; [ reflexivity | lia ]. }
      (* Disequalities. *)
      assert (Hwlt : ∀ x, x ∈ w :: nw :: map snd rest' ->
                       made_lt x (acc_next a)) by exact Hws.
      assert (HElt : ∀ x, x ∈ exn :: called :: returns ->
                       made_lt x (acc_next a))
        by (intros x Hx; exact (proj1 (HE x Hx))).
      assert (Hnotv : ∀ x, made_lt x (acc_next a) -> x <> v).
      { intros x Hx ->. specialize (Hx _ eq_refl). lia. }
      assert (Hfr : ∀ x, fresh_in (acc_next c) (acc_next a') x ->
                      made_lt x (acc_next a) -> False).
      { intros x Hf Hx. apply (fresh_in_lt (acc_next a) (acc_next a') x);
          [ | exact Hx ].
        apply (Hwid _ Hf). }
      assert (HnwE : ∀ x, x ∈ exn :: called :: returns -> x <> nw).
      { intros x Hx <-. apply (HEr x Hx). left. reflexivity. }
      exists (D' ++ F3 ++ F1).
      rewrite HD. cbn [c acc_facts].
      split; [ rewrite !app_assoc; reflexivity | ].
      split; [ exact Hcd | ]. split; [ exact Had | ].
      split; [ exact Hcl | ]. split; [ rewrite Hca in Hn; lia | ].
      assert (HF3i : ∀ f x, f ∈ F3 -> x ∈ into_nodes f -> x = v ∨ x = nw).
      { intros f x Hf Hx. unfold F3 in Hf. fsplit Hf; cbn in Hx;
          destruct Hx as [<- | []]; auto. }
      split; [ | split; [ | split; [ | split ] ] ].
      * intros x n Hf. apply in_app_or in Hf as [Hf | Hf].
        -- destruct (HD1 x n Hf) as (Hux & (w' & Hun) & Hxt & Hnf).
           cbn [map tl] in Hxt.
           assert (Hxa : made_lt x (acc_next a))
             by (apply Hwlt; right; right; exact Hxt).
           assert (Hxe : x ∉ exn :: called :: returns).
           { intros Hx. apply (HE x Hx). right; right. exact Hxt. }
           assert (Hnf' : fresh_in (acc_next c) (acc_next a') n) by exact Hnf.
           split; [ | split; [ | split ] ].
           ++ apply uniq_into_app; [ exact Hux | ].
              intros f Hf' Hx. apply in_app_or in Hf' as [Hf' | Hf'].
              ** destruct (HF3i _ _ Hf' Hx) as [-> | ->];
                   [ exact (Hnotv _ Hxa eq_refl) | exact (Hnwn Hxt) ].
              ** destruct (HF1i _ _ Hf' Hx) as [-> | ->];
                   apply Hxe; [ left | right; left ]; reflexivity.
           ++ exists w'. apply uniq_into_app; [ exact Hun | ].
              intros f Hf' Hx. apply in_app_or in Hf' as [Hf' | Hf'].
              ** destruct (HF3i _ _ Hf' Hx) as [-> | ->].
                 --- destruct Hnf' as (k' & Hk' & H1 & _). unfold v in Hk'.
                     injection Hk' as Hk'. rewrite Hca in H1. lia.
                 --- apply (Hfr nw Hnf'). apply Hwlt. right; left. reflexivity.
              ** destruct (HF1i _ _ Hf' Hx) as [-> | ->];
                   apply (Hfr _ Hnf'); apply HElt; [ left | right; left ];
                   reflexivity.
           ++ cbn [map tl]. right. exact Hxt.
           ++ apply (Hwid _ Hnf).
        -- apply in_app_or in Hf as [Hf | Hf]; [ | destruct (HF1u _ _ Hf) ].
           unfold F3 in Hf. fsplit Hf; try discriminate.
           injection Hf as <- <-.
           split; [ | split; [ | split ] ].
           ++ apply uniq_into_app_r.
              ** apply uniq_into_app;
                   [ split; [ right; left; reflexivity | ] | ].
                 --- intros f Hf Hx. unfold F3 in Hf. fsplit Hf; cbn in Hx;
                       destruct Hx as [Hx | []]; [ | reflexivity ].
                     exfalso. apply (Hnotv nw); [ apply Hwlt; right; left;
                                                  reflexivity | ].
                     symmetry. exact Hx.
                 --- intros f Hf Hx. destruct (HF1i _ _ Hf Hx) as [Hx' | Hx'];
                       refine (HnwE _ _ (eq_sym Hx'));
                       [ left | right; left ]; reflexivity.
              ** intros f Hf Hx. destruct (HD2 f nw Hf Hx) as [H | [H | H]].
                 --- apply (HEr nw H). left. reflexivity.
                 --- apply (Hfr nw H). apply Hwlt. right; left. reflexivity.
                 --- cbn [map tl] in H. exact (Hnwn H).
           ++ exists w. apply uniq_into_app_r.
              ** apply uniq_into_app; [ split; [ left; reflexivity | ] | ].
                 --- intros f Hf Hx. unfold F3 in Hf. fsplit Hf; cbn in Hx;
                       destruct Hx as [Hx | []]; [ reflexivity | ].
                     exfalso. apply (Hnotv nw); [ apply Hwlt; right; left;
                                                  reflexivity | exact Hx ].
                 --- intros f Hf Hx. destruct (HF1i _ _ Hf Hx) as [Hx' | Hx'];
                       [ apply (Hnotv exn); [ apply HElt; left; reflexivity
                                            | exact (eq_sym Hx') ]
                       | apply (Hnotv called);
                         [ apply HElt; right; left; reflexivity
                         | exact (eq_sym Hx') ] ].
              ** intros f Hf Hx. destruct (HD2 f v Hf Hx) as [H | [H | H]].
                 --- exact (Hnotv v (HElt v H) eq_refl).
                 --- destruct H as (k' & Hk' & H1 & _). unfold v in Hk'.
                     injection Hk' as Hk'. rewrite Hca in H1. lia.
                 --- cbn [map tl] in H.
                     exact (Hnotv v (Hwlt v (or_intror (or_intror H)))
                              eq_refl).
           ++ cbn [map tl]. left. reflexivity.
           ++ exact Hvf.
      * intros f x Hf Hx. apply in_app_or in Hf as [Hf | Hf].
        -- destruct (HD2 f x Hf Hx) as [H | [H | H]]; [ left; exact H | | ].
           ++ right; left. apply (Hwid _ H).
           ++ right; right. cbn [map tl] in H |- *. right. exact H.
        -- apply in_app_or in Hf as [Hf | Hf].
           ++ destruct (HF3i _ _ Hf Hx) as [-> | ->];
                [ right; left; exact Hvf | right; right; left; reflexivity ].
           ++ left. destruct (HF1i _ _ Hf Hx) as [-> | ->];
                [ left | right; left ]; reflexivity.
      * intros v' y [Hf | [c' Hf]]; apply in_app_or in Hf as [Hf | Hf];
          try (apply (HD3 v' y); first [ left; exact Hf
                                        | right; exists c'; exact Hf ]);
          apply in_app_or in Hf as [Hf | Hf];
          try (unfold F3 in Hf; fsplit Hf; discriminate);
          apply (HF1a v' y); first [ left; exact Hf
                                   | right; exists c'; exact Hf ].
      * intros c' y Hf. apply in_app_or in Hf as [Hf | Hf].
        -- right. exact (HD4 _ _ Hf).
        -- apply in_app_or in Hf as [Hf | Hf];
             [ unfold F3 in Hf; fsplit Hf; discriminate | ].
           left. symmetry. exact (HF1c _ _ Hf).
      * intros f x Hf Hx. apply in_app_or in Hf as [Hf | Hf].
        -- destruct (HD5 f x Hf Hx) as [H | [H | [H | H]]].
           ++ left. exact H.
           ++ right; left. apply (Hwid _ H).
           ++ right; right; left. right. exact H.
           ++ right; right; right. exact H.
        -- apply in_app_or in Hf as [Hf | Hf].
           ++ unfold F3 in Hf. fsplit Hf; cbn in Hx;
                repeat destruct Hx as [<- | Hx]; try contradiction;
                first [ right; left; exact Hvf
                      | right; right; left; left; reflexivity
                      | right; right; left; right; left; reflexivity ].
           ++ destruct (HF1n _ _ Hf Hx) as [-> | [-> | [-> | H]]].
              ** left. left. reflexivity.
              ** left. right; left. reflexivity.
              ** right; right; left. left. reflexivity.
              ** right; right; right. exact H.
Qed.

Lemma fresh_list_length (n : nat) (a : acc) :
  length (fst (fresh_list n a)) = n.
Proof.
  revert a. induction n as [ | n IH]; intros a; [ reflexivity | ].
  cbn [fresh_list fresh]. cbv beta iota.
  match goal with
  | |- context [fresh_list n ?b] =>
    specialize (IH b); destruct (fresh_list n b) as [xs c]
  end.
  cbn in *. rewrite IH. reflexivity.
Qed.

Lemma fresh_list_nodup (n : nat) (a : acc) : NoDup (fst (fresh_list n a)).
Proof.
  revert a. induction n as [ | n IH]; intros a; [ constructor | ].
  cbn [fresh_list fresh]. cbv beta iota.
  match goal with
  | |- context [fresh_list n ?b] =>
    specialize (IH b);
    destruct (AW.fresh_list_spec n b) as (_ & _ & _ & _ & _ & H6);
    destruct (fresh_list n b) as [xs c]
  end.
  cbn [fst snd acc_next] in *. constructor; [ | exact IH ].
  intros Hin. destruct (H6 _ Hin) as (m & Hm & H1 & _).
  injection Hm as <-. lia.
Qed.

Lemma map_snd_combine {A B : Type} (l1 : list A) (l2 : list B) :
  length l1 = length l2 -> map snd (combine l1 l2) = l2.
Proof.
  revert l2. induction l1 as [ | x l1 IH]; intros [ | y l2] H;
    cbn in *; try discriminate; [ reflexivity | ].
  rewrite IH; [ reflexivity | lia ].
Qed.

(* The call of unknown arity: the reads of its chain are made by it, and
   checked in its facts alone. *)
Lemma uinv_call_unknown (S : node -> Prop) (env : traverse_env)
    (ap : apply_expr) (exn_arg : node) (return_args : list node) (a : acc) :
  ap_call_kind ap = Indirect_unknown_arity ->
  uinv S a -> (∀ x, x ∈ exn_arg :: return_args -> good S a x) ->
  uinv (fun y => S y ∨ fresh_in (acc_next a)
                   (acc_next (traverse_call_kind env ap exn_arg return_args a))
                   y)
    (traverse_call_kind env ap exn_arg return_args a)
  ∧ (acc_next a
     <= acc_next (traverse_call_kind env ap exn_arg return_args a))%positive.
Proof.
  intros Hkind Hinv Hgood. unfold traverse_call_kind. rewrite Hkind.
  cbv zeta.
  destruct (AW.note_simple_fields (ap_callee ap) a) as (N1 & N2 & N3 & N4 & N5).
  set (closure := node_of_simple (ap_callee ap)).
  set (a1 := note_simple (ap_callee ap) a) in *. clearbody a1.
  unfold make_unknown_arity_apply_widget. cbn [fresh]. cbv beta iota zeta.
  set (called := Node_made (acc_next a1)).
  set (b1 := add_fact (Fact_any_usage called) _).
  set (groups := at_least_one_group _).
  assert (Hb1 : acc_next b1 = Pos.succ (acc_next a))
    by (cbn; rewrite N1; reflexivity).
  destruct (AW.fresh_list_spec (length groups) b1)
    as (F1 & F2 & F3 & F4 & F5 & F6).
  pose proof (fresh_list_nodup (length groups) b1) as Hnd.
  pose proof (fresh_list_length (length groups) b1) as Hlen.
  destruct (fresh_list (length groups) b1) as [ws b2] eqn:Ews.
  cbn [fst snd] in *.
  destruct (AW.note_simples_fields (ap_args ap) b2) as (M1 & M2 & M3 & M4 & M5).
  set (b3 := note_simples (ap_args ap) b2) in *. clearbody b3.
  destruct ws as [ | first rest].
  { exfalso. unfold groups, at_least_one_group in Hlen.
    destruct (group_by_parameter _ _); discriminate. }
  set (p := acc_next a) in *.
  assert (Hcalled : called = Node_made p)
    by (unfold called; rewrite N1; reflexivity).
  clearbody called. subst called.
  assert (Hws : ∀ w, w ∈ first :: rest ->
                  ∃ m, w = Node_made m ∧ (Pos.succ p <= m)%positive
                       ∧ (m < acc_next b2)%positive).
  { intros w Hw. destruct (F6 w Hw) as (m & -> & H1 & H2). exists m.
    rewrite Hb1 in H1. auto. }
  assert (Hsnd : map snd (combine groups (first :: rest)) = first :: rest)
    by (apply map_snd_combine; symmetry; exact Hlen).
  assert (HE : ∀ x, x ∈ exn_arg :: return_args -> made_lt x p ∧ ~ S x)
    by exact Hgood.
  assert (Hsep : sep_ok (exn_arg :: Node_made p :: return_args)
                   (map snd (combine groups (first :: rest))) (acc_next b3)).
  { rewrite Hsnd. split; [ exact Hnd | split ].
    - intros w Hw. destruct (Hws w Hw) as (m & -> & _ & H2).
      intros k Hk0. injection Hk0 as <-. rewrite M1. exact H2.
    - intros x Hx. split.
      + intros k Hk0. rewrite M1.
        destruct Hx as [<- | [<- | Hx]].
        * pose proof (proj1 (HE _ (or_introl eq_refl)) k Hk0). lia.
        * injection Hk0 as <-. lia.
        * pose proof (proj1 (HE _ (or_intror Hx)) k Hk0). lia.
      + intros Hw. destruct (Hws x Hw) as (m & -> & H1 & _).
        destruct Hx as [Hx | [Hx | Hx]].
        * pose proof (proj1 (HE _ (or_introl eq_refl)) m Hx). lia.
        * injection Hx as ->. lia.
        * pose proof (proj1 (HE _ (or_intror Hx)) m eq_refl). lia. }
  destruct (deps_spec return_args exn_arg (Node_made p)
              (combine groups (first :: rest)) b3 Hsep)
    as (D & HD & Hcd & Had & Hcl & Hn4 & HD1 & HD2 & HD3 & HD4 & HD5).
  rewrite Hsnd in HD1, HD2, HD4, HD5.
  set (b4 := unknown_arity_apply_deps return_args exn_arg (Node_made p)
               (combine groups (first :: rest)) b3) in *.
  clearbody b4. cbv beta iota.
  set (an := Node_made (acc_next b4)).
  set (CA := match env_current_code_id env with
             | None => Fact_alias first an
             | Some id => Fact_propagate (Node_code_id id) first an
             end).
  set (b5 := Acc (Pos.succ (acc_next b4)) _ _ _ _ _).
  assert (Ea2 : acc_facts (cond_alias env an first b5) = CA :: acc_facts b4
                ∧ acc_next (cond_alias env an first b5)
                  = Pos.succ (acc_next b4)
                ∧ acc_code_deps (cond_alias env an first b5) = acc_code_deps b4
                ∧ acc_apply_deps (cond_alias env an first b5)
                  = acc_apply_deps b4
                ∧ acc_closure_deps (cond_alias env an first b5)
                  = acc_closure_deps b4).
  { unfold cond_alias, CA. destruct (env_current_code_id env); cbn;
      repeat split. }
  destruct Ea2 as (Ef & En & Ec & Eap & Ecl).
  set (a2 := cond_alias env an first b5) in *. clearbody a2.
  set (au := Fact_any_usage (Node_made p)).
  set (Lw := Fact_accessor an unknown closure :: CA :: D ++ [au]).
  assert (Hfacts : acc_facts (add_fact (Fact_accessor an unknown closure) a2)
                   = Lw ++ acc_facts a).
  { cbn [add_fact add_facts acc_facts]. rewrite Ef, HD, M2, F1.
    unfold b1. cbn [add_fact add_facts acc_facts]. rewrite N2.
    unfold Lw. cbn [app]. rewrite <- app_assoc. reflexivity. }
  assert (Hnext : acc_next (add_fact (Fact_accessor an unknown closure) a2)
                  = Pos.succ (acc_next b4)) by exact En.
  assert (Hdeps : acc_code_deps (add_fact (Fact_accessor an unknown closure) a2)
                  = acc_code_deps a
                  ∧ acc_apply_deps
                      (add_fact (Fact_accessor an unknown closure) a2)
                    = acc_apply_deps a
                  ∧ acc_closure_deps
                      (add_fact (Fact_accessor an unknown closure) a2)
                    = acc_closure_deps a).
  { cbn [add_fact add_facts acc_code_deps acc_apply_deps acc_closure_deps].
    rewrite Ec, Eap, Ecl, Hcd, Had, Hcl, M3, M4, M5, F2, F3, F4.
    unfold b1. cbn. rewrite N3, N4, N5. auto. }
  set (a' := add_fact (Fact_accessor an unknown closure) a2) in *.
  clearbody a'.
  assert (Hb2 : (Pos.succ p <= acc_next b2)%positive)
    by (rewrite <- Hb1; exact F5).
  assert (Hb4 : (acc_next b2 <= acc_next b4)%positive)
    by (rewrite <- M1; exact Hn4).
  rewrite Hnext. split; [ | lia ].
  destruct Hdeps as (Dc & Dap & Dcl).
  set (q := Pos.succ (acc_next b4)) in *.
  set (R := fresh_in p q).
  assert (Ran : R an)
    by (exists (acc_next b4); split; [ reflexivity | unfold q; lia ]).
  assert (Rws : ∀ w, w ∈ first :: rest -> R w).
  { intros w Hw. destruct (Hws w Hw) as (m & -> & H1 & H2).
    exists m. split; [ reflexivity | unfold q; lia ]. }
  assert (Rfr : ∀ x, fresh_in (acc_next b3) (acc_next b4) x -> R x).
  { intros x (m & -> & H1 & H2). exists m.
    split; [ reflexivity | rewrite M1 in H1; unfold q; lia ]. }
  assert (Rp : R (Node_made p))
    by (exists p; split; [ reflexivity | unfold q; lia ]).
  assert (HEp : ∀ x, x ∈ exn_arg :: Node_made p :: return_args ->
                  made_lt x (Pos.succ p)).
  { intros x [<- | [<- | Hx]] k Hk.
    - pose proof (proj1 (HE _ (or_introl eq_refl)) k Hk). lia.
    - injection Hk as <-. lia.
    - pose proof (proj1 (HE _ (or_intror Hx)) k Hk). lia. }
  destruct (Hws first (or_introl eq_refl)) as (m0 & Hm0 & Hm1 & Hm2).
  apply NoDup_cons_iff in Hnd as [Hfr Hnd'].
  assert (Hne_an : ∀ x, made_lt x (acc_next b4) -> x <> an).
  { intros x Hx ->. specialize (Hx _ eq_refl). lia. }
  assert (Hne_first : ∀ x, made_lt x m0 ->
                        x <> first).
  { intros x Hx ->. specialize (Hx _ Hm0). lia. }
  assert (Hws_lt : ∀ w, w ∈ first :: rest -> made_lt w (acc_next b4)).
  { intros w Hw k Hk. destruct (Hws w Hw) as (m & Hm & _ & H2).
    rewrite Hm in Hk. injection Hk as <-. lia. }
  assert (Hfr_lt : ∀ x, fresh_in (acc_next b3) (acc_next b4) x ->
                     made_lt x (acc_next b4)).
  { intros x (m & -> & _ & H2) k Hk. injection Hk as <-. exact H2. }
  assert (HE_lt : ∀ x, x ∈ exn_arg :: Node_made p :: return_args ->
                    made_lt x m0).
  { intros x Hx k Hk. pose proof (HEp x Hx k Hk). lia. }
  assert (Hfr_ne_first : ∀ x, fresh_in (acc_next b3) (acc_next b4) x ->
                           x <> first).
  { intros x (m & -> & H1 & _) Hm. rewrite Hm0 in Hm. injection Hm as ->.
    rewrite M1 in H1. lia. }
  assert (HLi : ∀ f x, f ∈ Lw -> x ∈ into_nodes f ->
                  (f = Fact_accessor an unknown closure ∧ x = an)
                  ∨ (f = CA ∧ x = first)
                  ∨ (f ∈ D ∧ (x ∈ exn_arg :: Node_made p :: return_args
                              ∨ fresh_in (acc_next b3) (acc_next b4) x
                              ∨ x ∈ rest))).
  { intros f x Hf Hx. destruct Hf as [<- | [<- | Hf]].
    - left. cbn in Hx. destruct Hx as [<- | []]. auto.
    - right; left. split; [ reflexivity | ].
      unfold CA in Hx. destruct (env_current_code_id env); cbn in Hx;
        destruct Hx as [<- | []]; reflexivity.
    - apply in_app_or in Hf as [Hf | Hf].
      + right; right. split; [ exact Hf | ]. exact (HD2 f x Hf Hx).
      + destruct Hf as [<- | []]. destruct Hx. }
  assert (HLu : ∀ x n, Fact_accessor x unknown n ∈ Lw -> chain R Lw x n).
  { intros x n Hf.
    destruct Hf as [Hf | [Hf | Hf]].
    - injection Hf as <- <-.
      split; [ exact Ran | split; [ split; [ left; reflexivity | ] | split ] ].
      + intros f Hf Hx. destruct (HLi f an Hf Hx)
          as [[-> _] | [[_ H] | [_ [H | [H | H]]]]]; [ reflexivity | | | | ];
          exfalso.
        * exact (Hne_an first (Hws_lt first (or_introl eq_refl)) (eq_sym H)).
        * pose proof (HEp an H (acc_next b4) eq_refl). lia.
        * exact (Hne_an an (Hfr_lt an H) eq_refl).
        * exact (Hne_an an (Hws_lt an (or_intror H)) eq_refl).
      + left. intros c Hc. destruct Hc as [Hc | [Hc | Hc]].
        * discriminate.
        * unfold CA in Hc. destruct (env_current_code_id env); discriminate.
        * apply in_app_or in Hc as [Hc | Hc].
          -- exact (Hne_an an (Hws_lt an (HD4 _ _ Hc)) eq_refl).
          -- destruct Hc as [Hc | []]. discriminate.
      + intros w Hw.
        assert (Hw' : w = first).
        { destruct Hw as [Hw | [c Hw]];
            (destruct Hw as [Hw | [Hw | Hw]];
             [ discriminate | | ]);
            try (unfold CA in Hw; destruct (env_current_code_id env);
                 first [ discriminate
                       | injection Hw; intros; subst; reflexivity ]);
            apply in_app_or in Hw as [Hw | Hw];
            try (destruct Hw as [Hw | []]; discriminate);
            exfalso; apply (HD3 w an);
            [ left; exact Hw | right; exists c; exact Hw ]. }
        subst w. split; [ exact (Rws first (or_introl eq_refl)) | ].
        intros f Hf Hx. destruct (HLi f first Hf Hx)
          as [[-> H] | [[-> _] | [_ [H | [H | H]]]]].
        * exfalso. exact (Hne_an first (Hws_lt first (or_introl eq_refl)) H).
        * unfold CA. destruct (env_current_code_id env);
            [ right; eexists; reflexivity | left; reflexivity ].
        * exfalso. exact (Hne_first first (HE_lt _ H) eq_refl).
        * exfalso. exact (Hfr_ne_first first H eq_refl).
        * exfalso. exact (Hfr H).
    - unfold CA in Hf. destruct (env_current_code_id env); discriminate.
    - apply in_app_or in Hf as [Hf | Hf];
        [ | destruct Hf as [Hf | []]; discriminate ].
      destruct (HD1 x n Hf) as ((Hxin & Hxu) & (w & Hnin & Hnu) & Hxt & Hnf).
      cbn [tl] in Hxt.
      assert (Hxan : x <> an) by exact (Hne_an x (Hws_lt x (or_intror Hxt))).
      assert (Hxf : x <> first) by (intros ->; exact (Hfr Hxt)).
      assert (Hnan : n <> an) by exact (Hne_an n (Hfr_lt n Hnf)).
      assert (Hnf' : n <> first) by exact (Hfr_ne_first n Hnf).
      split; [ exact (Rws x (or_intror Hxt)) | split; [ split | split ] ].
      + right; right. apply in_or_app. left. exact Hxin.
      + intros f Hf' Hx. destruct (HLi f x Hf' Hx)
          as [[_ H] | [[_ H] | [Hf2 _]]];
          [ destruct (Hxan H) | destruct (Hxf H) | exact (Hxu f Hf2 Hx) ].
      + right. split; [ exact (Rfr n Hnf) | ]. exists w. split.
        * right; right. apply in_or_app. left. exact Hnin.
        * intros f Hf' Hx. destruct (HLi f n Hf' Hx)
            as [[_ H] | [[_ H] | [Hf2 _]]];
            [ destruct (Hnan H) | destruct (Hnf' H) | exact (Hnu f Hf2 Hx) ].
      + intros w' Hw'. exfalso.
        destruct Hw' as [Hw' | [c Hw']];
          (destruct Hw' as [Hw' | [Hw' | Hw']]; [ discriminate | | ]);
          try (unfold CA in Hw'; destruct (env_current_code_id env);
               first [ discriminate
                     | injection Hw'; intros; subst; exact (Hxan eq_refl) ]);
          apply in_app_or in Hw' as [Hw' | Hw'];
          try (destruct Hw' as [Hw' | []]; discriminate);
          apply (HD3 w' x); [ left; exact Hw' | right; exists c; exact Hw' ]. }
  (* The nodes of [Lw]. *)
  assert (HLn : ∀ f x, f ∈ Lw -> x ∈ nodes_of f ->
                  R x ∨ x ∈ exn_arg :: return_args ∨ not_made x).
  { intros f x Hf Hx. destruct Hf as [<- | [<- | Hf]].
    - cbn in Hx. destruct Hx as [<- | [<- | []]]; [ left; exact Ran | ].
      right; right. apply AW.not_made_simple.
    - unfold CA in Hx. destruct (env_current_code_id env); cbn in Hx;
        repeat destruct Hx as [<- | Hx]; try contradiction;
        try (left; first [ exact Ran | exact (Rws first (or_introl eq_refl)) ]);
        right; right; intros k; discriminate.
    - apply in_app_or in Hf as [Hf | Hf].
      + destruct (HD5 f x Hf Hx) as [[<- | [<- | H]] | [H | [H | H]]].
        * right; left. left. reflexivity.
        * left. exact Rp.
        * right; left. right. exact H.
        * left. exact (Rfr x H).
        * left. exact (Rws x H).
        * right; right. exact H.
      + destruct Hf as [<- | []]. cbn in Hx. destruct Hx as [<- | []].
        left. exact Rp. }
  assert (HSlt : ∀ y, S y -> ∃ k, y = Node_made k ∧ (k < p)%positive)
    by exact (ui_special S a Hinv).
  assert (HgoodS : ∀ x, good S a x ->
                     good (fun y => S y ∨ R y) a' x).
  { intros x [Hlt Hs]. split.
    - intros k Hk. specialize (Hlt k Hk). rewrite Hnext. unfold q in *. lia.
    - intros [H | H]; [ exact (Hs H) | ].
      destruct H as (k & -> & H1 & _). specialize (Hlt k eq_refl).
      unfold p in H1. lia. }
  constructor.
  - intros fct x Hin Hx. rewrite Hfacts in Hin.
    apply in_app_or in Hin as [Hin | Hin].
    + destruct (HLn fct x Hin Hx) as [H | [H | H]].
      * intros k Hk. destruct H as (k' & Hk' & _ & H2). rewrite Hk in Hk'.
        injection Hk' as ->. rewrite Hnext. exact H2.
      * intros k Hk. pose proof (proj1 (HE x H) k Hk). rewrite Hnext.
        unfold q, p in *. lia.
      * intros k Hk. destruct (H k Hk).
    + intros k Hk. pose proof (ui_nodes S a Hinv fct x Hin Hx k Hk).
      rewrite Hnext. unfold q, p in *. lia.
  - intros x [H | H].
    + destruct (HSlt x H) as (k & -> & Hk). exists k.
      split; [ reflexivity | rewrite Hnext; unfold q; lia ].
    + destruct H as (k & -> & _ & H2). exists k.
      split; [ reflexivity | rewrite Hnext; exact H2 ].
  - intros x n Hin. rewrite Hfacts in Hin |- *.
    apply in_app_or in Hin as [Hin | Hin].
    + apply (chain_frame R); [ intros y Hy; right; exact Hy
                             | exact (HLu x n Hin) | ].
      intros f y Hf Hy (k & -> & H1 & _).
      pose proof (ui_nodes S a Hinv f _ Hf Hy k eq_refl). unfold p in H1. lia.
    + apply (chain_frame_l S); [ intros y Hy; left; exact Hy
                               | exact (ui_chains S a Hinv x n Hin) | ].
      intros f y Hf Hy Hs. destruct (HSlt y Hs) as (k & -> & Hk).
      destruct (HLn f _ Hf Hy) as [H | [H | H]].
      * destruct H as (k' & Hk' & H1 & _). injection Hk' as ->. lia.
      * exact (proj2 (HE _ H) Hs).
      * exact (H k eq_refl).
  - intros id cd x Hin Hx. rewrite Dc in Hin.
    exact (HgoodS x (ui_code_deps S a Hinv id cd x Hin Hx)).
  - intros d x Hin Hx. rewrite Dap in Hin.
    exact (HgoodS x (ui_apply_deps S a Hinv d x Hin Hx)).
  - intros d Hin. rewrite Dcl in Hin.
    exact (HgoodS _ (ui_closure_deps S a Hinv d Hin)).
Qed.

(* ------------------------------------------------------------------------ *)
(* The steps of the traversal                                               *)
(* ------------------------------------------------------------------------ *)

(* [G'] is [G] and nodes made from [a]. *)
Definition grows (G : node -> Prop) (a : acc) (G' : node -> Prop) (a' : acc)
    : Prop :=
  (acc_next a <= acc_next a')%positive
  ∧ ∀ y, G' y -> G y ∨ ∃ k, y = Node_made k ∧ (acc_next a <= k)%positive.

Lemma grows_trans (G G1 G2 : node -> Prop) (a a1 a2 : acc) :
  grows G a G1 a1 -> grows G1 a1 G2 a2 -> grows G a G2 a2.
Proof.
  intros [Hn1 Hg1] [Hn2 Hg2]. split; [ lia | ].
  intros y Hy. destruct (Hg2 y Hy) as [H | (k & -> & Hk)].
  - destruct (Hg1 y H) as [H' | (k & -> & Hk)]; [ left; exact H' | ].
    right. exists k. split; [ reflexivity | lia ].
  - right. exists k. split; [ reflexivity | lia ].
Qed.

Lemma grows_same (G : node -> Prop) (a a' : acc) :
  (acc_next a <= acc_next a')%positive -> grows G a G a'.
Proof. intros H. split; [ exact H | intros y Hy; left; exact Hy ]. Qed.

Lemma env_good_grows (G G' : node -> Prop) (a a' : acc)
    (env : traverse_env) :
  grows G a G' a' -> env_good G env a -> env_good G' env a'.
Proof.
  intros [Hn Hg] H k ps x Hk Hx. destruct (H k ps x Hk Hx) as [Hlt Hs].
  split.
  - intros m Hm. specialize (Hlt m Hm). lia.
  - intros Hs'. destruct (Hg x Hs') as [H1 | (m & -> & Hm)];
      [ exact (Hs H1) | ].
    specialize (Hlt m eq_refl). lia.
Qed.

Definition tstep (env : traverse_env) (a a' : acc) : Prop :=
  ∀ G, uinv G a -> env_good G env a -> ∃ G', uinv G' a' ∧ grows G a G' a'.

Lemma tstep_refl (env : traverse_env) (a : acc) : tstep env a a.
Proof.
  intros G Hinv _. exists G. split; [ exact Hinv | apply grows_same; lia ].
Qed.

Lemma tstep_trans (env : traverse_env) (a a1 a2 : acc) :
  tstep env a a1 -> tstep env a1 a2 -> tstep env a a2.
Proof.
  intros H1 H2 G Hinv Henv. destruct (H1 G Hinv Henv) as (G1 & Hinv1 & Hg1).
  destruct (H2 G1 Hinv1 (env_good_grows _ _ _ _ _ Hg1 Henv))
    as (G2 & Hinv2 & Hg2).
  exists G2. split; [ exact Hinv2 | exact (grows_trans _ _ _ _ _ _ Hg1 Hg2) ].
Qed.

Lemma tstep_ext (env : traverse_env) (a a' : acc) :
  (∀ G, uinv G a -> env_good G env a -> ext G a a a') -> tstep env a a'.
Proof.
  intros H G Hinv Henv. pose proof (H G Hinv Henv) as He. exists G.
  split; [ exact (ext_close G a a' Hinv He) | ].
  apply grows_same. exact (ext_next _ _ _ _ He).
Qed.

Lemma tstep_fold {A : Type} (env : traverse_env) (f : acc -> A -> acc)
    (l : list A) (a : acc) :
  (∀ y b, y ∈ l -> tstep env b (f b y)) -> tstep env a (List.fold_left f l a).
Proof.
  revert a. induction l as [ | y l IH]; intros a H; cbn [fold_left].
  - apply tstep_refl.
  - apply (tstep_trans _ _ (f a y)); [ apply H; left; reflexivity | ].
    apply IH. intros y' b Hy'. apply H. right. exact Hy'.
Qed.

Lemma tstep_env (env env' : traverse_env) (a a' : acc) :
  (∀ G, uinv G a -> env_good G env a -> env_good G env' a) ->
  tstep env' a a' -> tstep env a a'.
Proof. intros H H' G Hinv Henv. exact (H' G Hinv (H G Hinv Henv)). Qed.

Ltac nexts' :=
  repeat match goal with
  | H : ext _ _ ?b ?c |- _ =>
    lazymatch goal with
    | _ : (acc_next b <= acc_next c)%positive |- _ => fail
    | _ => pose proof (ext_next _ _ _ _ H)
    end
  end.

Lemma uinv_call_kind (G : node -> Prop) (env : traverse_env)
    (ap : apply_expr) (exn_arg : node) (return_args : list node) (a : acc) :
  uinv G a -> (∀ x, x ∈ exn_arg :: return_args -> good G a x) ->
  ∃ G', uinv G' (traverse_call_kind env ap exn_arg return_args a)
        ∧ grows G a G' (traverse_call_kind env ap exn_arg return_args a).
Proof.
  intros Hinv Hgood.
  assert (Hdec : ap_call_kind ap = Indirect_unknown_arity
                 ∨ ap_call_kind ap <> Indirect_unknown_arity)
    by (destruct (ap_call_kind ap);
        (left; reflexivity) || (right; discriminate)).
  destruct Hdec as [Hk | Hk].
  - destruct (uinv_call_unknown G env ap exn_arg return_args a Hk Hinv Hgood)
      as [H1 H2].
    eexists. split; [ exact H1 | ]. split; [ exact H2 | ].
    intros y [Hy | (k & -> & H3 & _)]; [ left; exact Hy | ].
    right. exists k. auto.
  - pose proof (ext_call_known G a env ap exn_arg return_args a Hk
                  (Pos.le_refl _)
                  (fun x Hx => ok_good G a a x (Hgood x Hx))) as He.
    exists G. split; [ exact (ext_close G a _ Hinv He) | ].
    apply grows_same. exact (ext_next _ _ _ _ He).
Qed.

Lemma tstep_traverse_apply (env : traverse_env) (ap : apply_expr) (a : acc) :
  tstep env a (traverse_apply env ap a).
Proof.
  intros G Hinv Henv. unfold traverse_apply.
  destruct (ext_find_cont G a env (ap_continuation ap)
              (length (unarize (ap_return_arity ap))) a Henv)
    as [H1 H1o]; [ lia | ].
  destruct (find_cont env (ap_continuation ap) _ a) as [rets b1] eqn:E1.
  cbn [fst snd] in *. cbv beta iota.
  destruct (ext_find_cont G a env (ap_exn_continuation ap)
              (S (length (ap_exn_extra_args ap))) b1 Henv)
    as [H2 H2o]; [ nexts'; lia | ].
  destruct (find_cont env (ap_exn_continuation ap) _ b1) as [eps b2] eqn:E2.
  cbn [fst snd] in *. cbv beta iota.
  assert (Hpre : ∀ (exn_arg : node) (b3 : acc),
            ext G a b2 b3 -> ok_node G a b3 exn_arg ->
            ∃ G', uinv G' (traverse_call_kind env ap exn_arg rets b3)
                  ∧ grows G a G' (traverse_call_kind env ap exn_arg rets b3)).
  { intros exn_arg b3 H3 Hex.
    assert (H03 : ext G a a b3)
      by exact (ext_trans _ _ _ _ _ (ext_trans _ _ _ _ _ H1 H2) H3).
    pose proof (ext_close G a b3 Hinv H03) as Hinv3.
    pose proof (ext_next _ _ _ _ H03) as Hn3.
    destruct (uinv_call_kind G env ap exn_arg rets b3 Hinv3)
      as (G' & Hinv4 & Hg4).
    - intros x [<- | Hx]; apply (ok_close G a b3); try assumption.
      pose proof (H1o x Hx). nexts'. eapply ok_node_mono; [ | eassumption ].
      lia.
    - exists G'. split; [ exact Hinv4 | ].
      exact (grows_trans _ _ _ _ _ _ (grows_same G a b3 Hn3) Hg4). }
  destruct eps as [ | ep extra].
  - destruct (fresh b2) as [x b3] eqn:E3. cbv beta iota.
    destruct (fresh_spec G a b2 x b3 E3) as (_ & _ & He & Ho);
      [ nexts'; lia | ].
    exact (Hpre x b3 He Ho).
  - cbv zeta beta iota.
    pose proof (ext_note_simples G a (exn_extra_simples ap) b2) as H3.
    destruct (AW.note_simples_fields (exn_extra_simples ap) b2) as (N1 & _).
    set (b3 := note_simples _ b2) in *. clearbody b3.
    apply Hpre.
    + apply (ext_trans _ _ _ _ _ H3). apply ext_add_facts.
      * intros fct y H Hy. apply in_map_iff in H.
        destruct H as [[p s] [<- Hc]]. cbn in Hy.
        destruct Hy as [<- | [<- | []]]; [ | apply ok_simple ].
        pose proof (H2o p (or_intror (in_combine_l _ _ _ _ Hc))).
        nexts'. eapply ok_node_mono; [ | eassumption ]. lia.
      * intros w n H. apply in_map_iff in H.
        destruct H as [[p s] [H _]]. discriminate.
    + pose proof (H2o ep (or_introl eq_refl)). nexts'.
      eapply ok_node_mono; [ | eassumption ]. cbn [acc_next add_facts]. lia.
Qed.

Lemma good_not_made (G : node -> Prop) (a : acc) (x : node) :
  uinv G a -> not_made x -> good G a x.
Proof.
  intros Hinv Hx. split; [ intros k -> ; destruct (Hx k eq_refl) | ].
  intros Hs. destruct (ui_special G a Hinv x Hs) as (k & -> & _).
  exact (Hx k eq_refl).
Qed.

Lemma env_good_add_cont (G : node -> Prop) (env : traverse_env)
    (k : continuation) (params : list node) (a : acc) :
  env_good G env a -> (∀ x, x ∈ params -> good G a x) ->
  env_good G (add_cont env k params) a.
Proof.
  intros Henv Hp k' params' x [Heq | Hin] Hx.
  - injection Heq as <- <-. exact (Hp x Hx).
  - exact (Henv _ _ _ Hin Hx).
Qed.

Lemma handler_params_good (G : node -> Prop) (a : acc)
    (h : continuation_handler) :
  uinv G a -> ∀ x, x ∈ handler_params h -> good G a x.
Proof.
  intros Hinv x Hx. destruct h as [params md body]. cbn in Hx.
  apply in_map_iff in Hx. destruct Hx as [y [<- _]].
  apply good_not_made; [ exact Hinv | intros k; discriminate ].
Qed.

Lemma env_good_add_rec_conts (G : node -> Prop) (env : traverse_env)
    (handlers : list (continuation * continuation_handler)) (a : acc) :
  uinv G a -> env_good G env a -> env_good G (add_rec_conts env handlers) a.
Proof.
  intros Hinv Henv. unfold add_rec_conts.
  induction handlers as [ | [k h] handlers IH]; cbn [fold_right];
    [ exact Henv | ].
  apply env_good_add_cont; [ exact IH | ].
  apply handler_params_good. exact Hinv.
Qed.

Lemma in_cd_exn (cd : code_dep) : cd_exn cd ∈ code_dep_nodes cd.
Proof. right; left. reflexivity. Qed.

Lemma in_cd_return (cd : code_dep) (x : node) :
  x ∈ cd_return cd -> x ∈ code_dep_nodes cd.
Proof.
  intros H. right; right; right. apply in_or_app. right. apply in_or_app.
  left. exact H.
Qed.

Lemma in_cd_params (cd : code_dep) (x : node) :
  x ∈ cd_params cd -> x ∈ code_dep_nodes cd.
Proof. intros H. right; right; right. apply in_or_app. left. exact H. Qed.

Lemma tstep_traverse :
  (∀ e env a, tstep env a (traverse env e a))
  ∧ (∀ h env a, tstep env a (traverse_handler env h a)).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - intros x kind d body IH env a. cbn [traverse].
    apply (tstep_trans _ _ (traverse_named env x d a)); [ | apply IH ].
    apply tstep_ext. intros G _ _. apply ext_traverse_named. lia.
  - intros bindings body IHc IH env a. cbn [traverse]. cbv zeta.
    eapply tstep_trans; [ | apply IH ].
    eapply tstep_trans; [ | apply tstep_fold ].
    + eapply tstep_trans; [ | apply tstep_fold ].
      * apply tstep_fold. intros b a' _. apply tstep_ext. intros G _ _.
        apply ext_prepare_binding. lia.
      * intros b a' _. apply tstep_ext. intros G _ _.
        apply ext_traverse_static_binding.
    + intros [id c | id | s const] a' Hin; cbn [traverse_binding];
        [ | apply tstep_refl | apply tstep_refl ].
      destruct c as [params mc ret exn md cbody]. cbn [traverse_code].
      destruct (lookup id (acc_code_deps a')) as [cd | ] eqn:Ecd;
        [ | apply tstep_refl ].
      set (b := add_facts _ a').
      apply (tstep_trans _ _ b).
      * apply tstep_ext. intros G Hinv _. apply ext_add_facts.
        -- assert (Hin' : (id, cd) ∈ acc_code_deps a')
             by exact (FlambdaMapProofs.lookup_in _ _ _ Ecd).
           intros fct y H Hy. destruct H as [<- | H].
           ++ cbn in Hy. destruct Hy as [<- | [<- | []]];
                [ apply ok_not_made; intros k; discriminate | ].
              apply ok_good. exact (ui_code_deps G a' Hinv id cd _ Hin'
                                      (in_cd_my cd)).
           ++ apply in_map_iff in H. destruct H as [[p arg] [<- Hc]].
              cbn in Hy. destruct Hy as [<- | [<- | []]];
                [ apply ok_not_made; intros k; discriminate | ].
              apply ok_good. apply (ui_code_deps G a' Hinv id cd _ Hin').
              apply in_cd_params. exact (in_combine_r _ _ _ _ Hc).
        -- intros w n H. destruct H as [H | H]; [ discriminate | ].
           apply in_map_iff in H. destruct H as [[p arg] [H _]].
           discriminate.
      * apply (tstep_env _ (code_env env id ret exn cd));
          [ | apply (IHc id (Code params mc ret exn md cbody) Hin) ].
        intros G Hinvb _ k ps y Hk Hy.
        assert (Hin' : (id, cd) ∈ acc_code_deps b)
          by exact (FlambdaMapProofs.lookup_in _ _ _ Ecd).
        cbn [code_env env_conts] in Hk.
        destruct Hk as [Hk | [Hk | []]]; injection Hk as <- <-;
          apply (ui_code_deps G b Hinvb id cd y Hin').
        -- destruct Hy as [<- | []]. apply in_cd_exn.
        -- apply in_cd_return. exact Hy.
  - intros k h body IHh IH env a. cbn [traverse]. cbv zeta.
    eapply tstep_trans; [ apply IHh | ].
    eapply tstep_trans; [ apply tstep_ext; intros G _ _;
                          apply ext_exn_handler_deps | ].
    apply (tstep_env _ (add_cont env k (handler_params h))); [ | apply IH ].
    intros G Hinv Henv. apply env_good_add_cont; [ exact Henv | ].
    apply handler_params_good. exact Hinv.
  - intros handlers body IHh IH env a. cbn [traverse]. cbv zeta.
    apply (tstep_env _ (add_rec_conts env handlers));
      [ intros G Hinv Henv; apply env_good_add_rec_conts; assumption | ].
    eapply tstep_trans; [ | apply IH ].
    apply tstep_fold. intros [k h] a' Hin. apply (IHh k h Hin).
  - intros ap env a. apply tstep_traverse_apply.
  - intros ac env a. apply tstep_ext. intros G _ Henv.
    apply ext_apply_cont_deps; [ exact Henv | lia ].
  - intros scrutinee arms env a. apply tstep_ext. intros G _ Henv.
    apply ext_traverse_switch; [ exact Henv | lia ].
  - intros env a. apply tstep_refl.
  - intros params md handler IH env a. cbn [traverse_handler].
    apply IH.
Qed.

(* ------------------------------------------------------------------------ *)
(* The unit                                                                 *)
(* ------------------------------------------------------------------------ *)

Lemma uinv_empty : uinv (fun _ => False) empty_acc.
Proof.
  constructor; cbn; try (intros; contradiction).
  intros x n [].
Qed.

Lemma uinv_fresh (G : node -> Prop) (a : acc) :
  uinv G a -> uinv G (snd (fresh a)) ∧ good G (snd (fresh a)) (fst (fresh a)).
Proof.
  intros Hinv. split; [ exact (ext_close G a _ Hinv (ext_fresh G a a)) | ].
  apply (ok_close G a); [ exact Hinv | cbn; lia | ].
  apply ok_fresh. lia.
Qed.

Lemma uinv_traverse_unit (flag : reaper_preserve_direct_calls)
    (u : unit_interface) (e : expr) :
  ∃ G, uinv G (traverse_unit_acc flag u e).
Proof.
  unfold traverse_unit_acc. cbv zeta.
  set (a1 := add_fact (Fact_any_source Node_all_constants) empty_acc).
  assert (Hinv1 : uinv (fun _ => False) a1).
  { apply (ext_close _ empty_acc); [ exact uinv_empty | ].
    apply ext_add_fact; [ | intros ? ? ?; discriminate ].
    intros y [<- | []]. apply ok_not_made. intros k; discriminate. }
  destruct (uinv_fresh _ a1 Hinv1) as (Hinv2 & Hr).
  destruct (fresh a1) as [r a2] eqn:E2. cbn [fst snd] in *.
  destruct (uinv_fresh _ a2 Hinv2) as (Hinv3 & Hex).
  destruct (fresh a2) as [ex a3] eqn:E3. cbn [fst snd] in *.
  cbv beta iota zeta.
  set (a4 := add_fact (Fact_any_usage ex) a3).
  assert (Hinv4 : uinv (fun _ => False) a4).
  { apply (ext_close _ a3); [ exact Hinv3 | ].
    apply ext_add_fact; [ | intros ? ? ?; discriminate ].
    intros y [<- | []]. apply ok_good. exact Hex. }
  match goal with
  | |- ∃ G, uinv G (resolve_delayed_deps _ (add_fact _ (traverse ?env e a4))) =>
    set (env0 := env)
  end.
  assert (Henv : env_good (fun _ => False) env0 a4).
  { intros k ps x Hk Hx. cbn [env0 env_conts] in Hk.
    destruct Hk as [Hk | [Hk | []]]; injection Hk as <- <-;
      destruct Hx as [<- | []].
    - apply (good_mono _ a3); [ cbn; lia | exact Hex ].
    - apply (good_mono _ a2); [ unfold a4; cbn;
        unfold fresh in E3; injection E3 as _ <-; cbn; lia | exact Hr ]. }
  destruct (proj1 tstep_traverse e env0 a4 _ Hinv4 Henv)
    as (G & Hinv5 & Hg5).
  pose proof (env_good_grows _ _ _ _ _ Hg5 Henv) as Henv5.
  set (a5 := traverse env0 e a4) in *.
  assert (Hinv6 : uinv G (add_fact (Fact_any_usage r) a5)).
  { apply (ext_close _ a5); [ exact Hinv5 | ].
    apply ext_add_fact; [ | intros ? ? ?; discriminate ].
    intros y [<- | []]. apply ok_good.
    apply (Henv5 (unit_return u) [r]); [ | left; reflexivity ].
    right; left. reflexivity. }
  exists G. apply (ext_close _ _ _ Hinv6). apply ext_resolve. exact Hinv6.
Qed.

Lemma traverse_unknown_chains (flag : reaper_preserve_direct_calls)
    (u : unit_interface) (e : expr) :
  check_unknown_chains (traverse_unit flag u e) = true.
Proof.
  rewrite traverse_unit_acc_facts.
  destruct (uinv_traverse_unit flag u e) as [G H].
  exact (check_of_chains G _ (ui_chains G _ H)).
Qed.

End Chains.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Theorem traverse_unknown_chains : traverse_unknown_chains_statement.
Proof. intros flag u e. exact (Chains.traverse_unknown_chains flag u e). Qed.

Theorem indirect_dead_frame_traversal :
  indirect_dead_frame_traversal_statement.
Proof.
  intros flag is_local u e T n s ap h vars conts recursive Hlocal Hslots
    Hfrag Hleast.
  exact (FlambdaReaperIndirectDeadFramesProofs.indirect_dead_frame flag
           is_local u e T n s ap h vars conts recursive Hlocal Hslots Hfrag
           Hleast (traverse_unknown_chains flag u e)).
Qed.

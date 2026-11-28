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

(* Proofs of the statements of FlambdaReaperRegionsSpec.v.

   This file needs no review: what it proves is stated there. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaPrimitiveModel
  FlambdaModel FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec
  FlambdaSemanticsSpec FlambdaWellFormedSpec FlambdaNameAbstractionSpec
  FlambdaRenameApartSpec FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperSolution FlambdaReaperSolutionSpec FlambdaReaperFlowSpec
  FlambdaReaperRebuild FlambdaReaperRebuildSpec FlambdaReaperLeastSpec
  FlambdaReaperLocalFieldsSpec FlambdaRegionsSpec FlambdaRegionsErasureSpec
  FlambdaRegionsCheckSpec FlambdaRegionsCheckSoundSpec
  FlambdaReaperRegionsSpec FlambdaReaperRegionsRebuildProofs
  FlambdaReaperRegionsTraverseProofs.
From OxCamlFlambda2 Require FlambdaSemanticsProofs
  FlambdaReaperSolutionProofs FlambdaReaperLeastProofs FlambdaReaperFlowProofs
  FlambdaReaperLocalFieldsProofs FlambdaRegionsErasureProofs
  FlambdaRegionsCheckSoundProofs FlambdaNameAbstractionProofs
  FlambdaRegionsCallsProofs FlambdaRegionsCheckProofs
  FlambdaReaperRegionsKeepsProofs FlambdaReaperRebuildProofs.
Import ListNotations.

(* ======================================================================== *)
(* The graph of the erased unit                                             *)
(* ======================================================================== *)

(* The graph [G'] of the erased unit is that of the unit, [G], but for
   facts of the regions that it has not, and an alias of a variable that
   has any source to the constants. *)
Record graph_below (G G' : graph) : Prop := {
  gb_alias : ∀ t f, g_alias G' t f ->
    g_alias G t f
    ∨ ∃ x, t = Node_variable x ∧ f = Node_all_constants ∧ g_any_source G t;
  gb_use : ∀ t f, g_use G' t f -> g_use G t f;
  gb_accessor : ∀ t r b, g_accessor G' t r b <-> g_accessor G t r b;
  gb_constructor : ∀ b r f, g_constructor G' b r f <-> g_constructor G b r f;
  gb_argument : ∀ f i b, g_argument G' f i b <-> g_argument G f i b;
  gb_parameter : ∀ b i t, g_parameter G' b i t <-> g_parameter G b i t;
  gb_propagate : ∀ i t f, g_propagate G' i t f -> g_propagate G i t f;
  gb_alias_if : ∀ i t f,
    g_alias_if_any_source G' i t f -> g_alias_if_any_source G i t f;
  gb_any_usage : ∀ x, g_any_usage G' x -> g_any_usage G x;
  gb_any_source : ∀ x, g_any_source G' x -> g_any_source G x;
  gb_any_source_if_used : ∀ i x,
    g_any_source_if_used G' i x -> g_any_source_if_used G i x;
  gb_zero : ∀ x, g_zero_alloc_source G' x -> g_zero_alloc_source G x
}.

(* No fact of [G] has the constants where something is put. *)
Record positions_ok (G : graph) : Prop := {
  po_alias : ∀ t f, g_alias G t f -> t <> Node_all_constants;
  po_use : ∀ t f, g_use G t f -> t <> Node_all_constants;
  po_accessor : ∀ t r b, g_accessor G t r b -> t <> Node_all_constants;
  po_constructor : ∀ b r f, g_constructor G b r f -> b <> Node_all_constants;
  po_parameter : ∀ b i t, g_parameter G b i t ->
    b <> Node_all_constants ∧ t <> Node_all_constants;
  po_propagate : ∀ i t f, g_propagate G i t f ->
    i <> Node_all_constants ∧ t <> Node_all_constants;
  po_alias_if : ∀ i t f, g_alias_if_any_source G i t f ->
    t <> Node_all_constants;
  po_any_source_if_used : ∀ i x, g_any_source_if_used G i x ->
    i <> Node_all_constants
}.

Lemma in_flat_map_one {B : Type} (p : fact -> list B) (l : list fact)
    (y : B) :
  y ∈ flat_map p l <-> ∃ fct, fct ∈ l ∧ y ∈ p fct.
Proof. apply in_flat_map. Qed.

Ltac fact_in H :=
  let fct := fresh "fct" in let Hf := fresh "Hf" in let Hy := fresh "Hy" in
  apply in_flat_map in H as [fct [Hf Hy]];
  destruct fct; cbn in Hy; try contradiction;
  destruct Hy as [Hy | []]; inversion Hy; subst; try clear Hy.

Lemma graph_positions (l : list fact) :
  (∀ fct, fct ∈ l -> pos_ok fct) ->
  positions_ok (graph_of_facts (facts_of_list l)).
Proof.
  intros H. constructor; cbn [graph_of_facts g_alias g_use g_accessor
    g_constructor g_parameter g_propagate g_alias_if_any_source
    g_any_source_if_used facts_of_list f_alias f_use f_accessor f_constructor
    f_parameter f_propagate f_alias_if_any_source f_any_source_if_used].
  - intros t f Hin. fact_in Hin. exact (H _ Hf).
  - intros t f Hin. fact_in Hin. exact (H _ Hf).
  - intros t r b Hin. fact_in Hin. exact (H _ Hf).
  - intros b r f Hin. fact_in Hin. exact (H _ Hf).
  - intros b i t Hin. fact_in Hin. exact (H _ Hf).
  - intros i t f Hin. fact_in Hin. exact (H _ Hf).
  - intros i t f Hin. fact_in Hin. exact (H _ Hf).
  - intros i x Hin. fact_in Hin. exact (H _ Hf).
Qed.

Ltac fact_out fct Hf :=
  apply in_flat_map; exists fct; split; [ exact Hf | left; reflexivity ].

Lemma graph_below_of_rel (l l' : list fact) :
  facts_rel l l' ->
  graph_below (graph_of_facts (facts_of_list l))
    (graph_of_facts (facts_of_list l')).
Proof.
  intros [(HC & HA & HP & HArg) Hi].
  constructor; cbn [graph_of_facts g_alias g_use g_accessor g_constructor
    g_argument g_parameter g_propagate g_alias_if_any_source g_any_usage
    g_any_source g_any_source_if_used g_zero_alloc_source].
  - intros t f Hin. fact_in Hin.
    destruct (Hi _ Hf) as [Hl | (x & Heq & Hx)].
    + left. fact_out (Fact_alias t f) Hl.
    + injection Heq as -> ->. right. exists x.
      split; [ reflexivity | split; [ reflexivity | ] ].
      fact_out (Fact_any_source (Node_variable x)) Hx.
  - intros t f Hin. fact_in Hin.
    destruct (Hi _ Hf) as [Hl | (x & Heq & _)]; [ | discriminate Heq ].
    fact_out (Fact_use t f) Hl.
  - intros t r b. rewrite HA. reflexivity.
  - intros b r f. rewrite HC. reflexivity.
  - intros f i b. rewrite HArg. reflexivity.
  - intros b i t. rewrite HP. reflexivity.
  - intros i t f Hin. fact_in Hin.
    destruct (Hi _ Hf) as [Hl | (x & Heq & _)]; [ | discriminate Heq ].
    fact_out (Fact_propagate i t f) Hl.
  - intros i t f Hin. fact_in Hin.
    destruct (Hi _ Hf) as [Hl | (x & Heq & _)]; [ | discriminate Heq ].
    fact_out (Fact_alias_if_any_source i t f) Hl.
  - intros x Hin. fact_in Hin.
    destruct (Hi _ Hf) as [Hl | (y & Heq & _)]; [ | discriminate Heq ].
    fact_out (Fact_any_usage x) Hl.
  - intros x Hin. fact_in Hin.
    destruct (Hi _ Hf) as [Hl | (y & Heq & _)]; [ | discriminate Heq ].
    fact_out (Fact_any_source x) Hl.
  - intros i x Hin. fact_in Hin.
    destruct (Hi _ Hf) as [Hl | (y & Heq & _)]; [ | discriminate Heq ].
    fact_out (Fact_any_source_if_used i x) Hl.
  - intros x [].
Qed.

(* ======================================================================== *)
(* The solution for the erased unit                                         *)
(* ======================================================================== *)

(* The solution [S] with the constants used everywhere, and every
   variable that has any source an alias of the constants. *)
Definition consts_used (S : solution) : solution :=
  {| s_alias := fun t f =>
       s_alias S t f
       || match t with
          | Node_variable _ => node_eqb f Node_all_constants && s_any_source S t
          | _ => false
          end;
     s_any_usage := fun n =>
       match n with
       | Node_all_constants => true
       | _ => s_any_usage S n
       end;
     s_any_source := s_any_source S;
     s_has_usage := fun n =>
       match n with
       | Node_all_constants => true
       | _ => s_has_usage S n
       end;
     s_has_source := s_has_source S;
     s_usages := s_usages S;
     s_sources := s_sources S;
     s_reading_field := s_reading_field S;
     s_escaping_field := s_escaping_field S;
     s_field_used := s_field_used S |}.

Lemma node_eqb_true (n n' : node) : node_eqb n n' = true -> n = n'.
Proof. unfold node_eqb. destruct (node_eq_dec n n'); [ auto | discriminate ].
Qed.

Lemma cu_alias (S : solution) (t f : node) :
  s_alias (consts_used S) t f = true ->
  s_alias S t f = true
  ∨ (∃ x, t = Node_variable x) ∧ f = Node_all_constants
    ∧ s_any_source S t = true.
Proof.
  cbn. intros H. apply orb_true_iff in H as [H | H]; [ left; exact H | ].
  right. destruct t; try discriminate H.
  apply andb_true_iff in H as [Hf Ha]. apply node_eqb_true in Hf.
  split; [ eexists; reflexivity | split; assumption ].
Qed.

Lemma cu_alias_old (S : solution) (t f : node) :
  s_alias S t f = true -> s_alias (consts_used S) t f = true.
Proof. intros H. cbn. rewrite H. reflexivity. Qed.

Lemma cu_usage (S : solution) (n : node) :
  n <> Node_all_constants ->
  s_any_usage (consts_used S) n = s_any_usage S n
  ∧ s_has_usage (consts_used S) n = s_has_usage S n.
Proof. intros H. destruct n; try (split; reflexivity). contradiction. Qed.

Lemma cu_any_usage_old (S : solution) (n : node) :
  s_any_usage S n = true -> s_any_usage (consts_used S) n = true.
Proof. destruct n; cbn; auto. Qed.

Lemma cu_has_usage_old (S : solution) (n : node) :
  s_has_usage S n = true -> s_has_usage (consts_used S) n = true.
Proof. destruct n; cbn; auto. Qed.

Lemma cu_any_usage_false (S : solution) (n : node) :
  s_any_usage (consts_used S) n = false ->
  s_any_usage S n = false ∧ n <> Node_all_constants.
Proof.
  destruct n; cbn; intros H; try (split; [ exact H | discriminate ]).
  discriminate H.
Qed.

Section Closed.

Variable is_local : field -> bool.
Variables G G' : graph.
Variable S : solution.
Hypothesis Hc : closed is_local G S.
Hypothesis Hna : ∀ n, s_alias S Node_all_constants n = false.
Hypothesis Hnr : ∀ r, s_reading_field S r Node_all_constants = false.
Hypothesis Hpos : positions_ok G.
Hypothesis Hgb : graph_below G G'.

Abbreviation S' := (consts_used S).

Lemma ne_ac_usage (n : node) :
  n <> Node_all_constants ->
  s_any_usage S' n = s_any_usage S n ∧ s_has_usage S' n = s_has_usage S n.
Proof. apply cu_usage. Qed.

Lemma consts_used_closed : closed is_local G' S'.
Proof.
  constructor.
  - intros t f Hg.
    destruct (gb_alias _ _ Hgb t f Hg) as [H | (x & -> & -> & Hx)].
    + apply cu_alias_old. exact (alias_input _ _ _ Hc t f H).
    + cbn. apply orb_true_iff. right. unfold node_eqb.
      destruct (node_eq_dec Node_all_constants Node_all_constants)
        as [_ | Hn]; [ | contradiction Hn; reflexivity ].
      exact (any_source_input _ _ _ Hc _ Hx).
  - intros i t f Hu Hg.
    pose proof (gb_propagate _ _ Hgb _ _ _ Hg) as HgG.
    destruct (po_propagate _ Hpos _ _ _ HgG) as [Hi _].
    rewrite (proj1 (ne_ac_usage i Hi)) in Hu. apply cu_alias_old.
    exact (alias_propagate _ _ _ Hc _ _ _ Hu HgG).
  - intros i t f Ha Hg. apply cu_alias_old.
    exact (alias_if_any_source _ _ _ Hc _ _ _ Ha
             (gb_alias_if _ _ Hgb _ _ _ Hg)).
  - intros base bu r t f Hcons Hus Hau Hacc. apply cu_alias_old.
    exact (alias_constructor_usage _ _ _ Hc base bu r t f
             (proj1 (gb_constructor _ _ Hgb _ _ _) Hcons) Hus
             (proj1 (cu_any_usage_false _ _ Hau))
             (proj1 (gb_accessor _ _ Hgb _ _ _) Hacc)).
  - intros base bu r t f Hl Hcons Hus Hacc. apply cu_alias_old.
    exact (alias_local_constructor_usage _ _ _ Hc base bu r t f Hl
             (proj1 (gb_constructor _ _ Hgb _ _ _) Hcons) Hus
             (proj1 (gb_accessor _ _ Hgb _ _ _) Hacc)).
  - intros base bu r t f Hpar Hus Hau Harg. apply cu_alias_old.
    exact (alias_parameter_usage _ _ _ Hc base bu r t f
             (proj1 (gb_parameter _ _ Hgb _ _ _) Hpar) Hus
             (proj1 (cu_any_usage_false _ _ Hau))
             (proj1 (gb_argument _ _ Hgb _ _ _) Harg)).
  - intros base bs r t f Hacc Hsrc Has Hcons. apply cu_alias_old.
    exact (alias_accessor_source _ _ _ Hc base bs r t f
             (proj1 (gb_accessor _ _ Hgb _ _ _) Hacc) Hsrc Has
             (proj1 (gb_constructor _ _ Hgb _ _ _) Hcons)).
  - intros base bs r t f Hl Hacc Hsrc Hcons. apply cu_alias_old.
    exact (alias_local_accessor_source _ _ _ Hc base bs r t f Hl
             (proj1 (gb_accessor _ _ Hgb _ _ _) Hacc) Hsrc
             (proj1 (gb_constructor _ _ Hgb _ _ _) Hcons)).
  - intros r t f He Hr. apply cu_alias_old.
    exact (alias_escaping_reading _ _ _ Hc r t f He Hr).
  - intros base bs r t f Harg Hsrc Has Hpar. apply cu_alias_old.
    exact (alias_argument_source _ _ _ Hc base bs r t f
             (proj1 (gb_argument _ _ Hgb _ _ _) Harg) Hsrc Has
             (proj1 (gb_parameter _ _ Hgb _ _ _) Hpar)).
  - intros x Hx.
    destruct (node_eq_dec x Node_all_constants) as [-> | Hne];
      [ reflexivity | ].
    rewrite (proj2 (ne_ac_usage x Hne)).
    rewrite (proj1 (ne_ac_usage x Hne)) in Hx.
    exact (has_usage_any _ _ _ Hc x Hx).
  - intros t f Hu Ha.
    destruct (cu_alias S t f Ha) as [Ha' | (_ & -> & _)]; [ | reflexivity ].
    destruct (node_eq_dec t Node_all_constants) as [-> | Hne];
      [ rewrite Hna in Ha'; discriminate Ha' | ].
    rewrite (proj2 (ne_ac_usage t Hne)) in Hu. apply cu_has_usage_old.
    exact (has_usage_alias _ _ _ Hc t f Hu Ha').
  - intros t r b Hu Hacc.
    pose proof (proj1 (gb_accessor _ _ Hgb t r b) Hacc) as HaccG.
    pose proof (po_accessor _ Hpos _ _ _ HaccG) as Hne.
    rewrite (proj2 (ne_ac_usage t Hne)) in Hu. apply cu_has_usage_old.
    exact (has_usage_accessor _ _ _ Hc t r b Hu HaccG).
  - intros f r b Hs Harg. apply cu_has_usage_old.
    exact (has_usage_argument _ _ _ Hc f r b Hs
             (proj1 (gb_argument _ _ Hgb _ _ _) Harg)).
  - intros x H. exact (has_source_any _ _ _ Hc x H).
  - intros f t Hs Ha.
    destruct (cu_alias S t f Ha) as [Ha' | (_ & _ & Has)];
      [ exact (has_source_alias _ _ _ Hc f t Hs Ha')
      | exact (has_source_any _ _ _ Hc t Has) ].
  - intros f r b Hs Hcons.
    exact (has_source_constructor _ _ _ Hc f r b Hs
             (proj1 (gb_constructor _ _ Hgb _ _ _) Hcons)).
  - intros f r b Hcons Hfs.
    exact (has_source_function_slot _ _ _ Hc f r b
             (proj1 (gb_constructor _ _ Hgb _ _ _) Hcons) Hfs).
  - intros t r b Hu Hpar.
    pose proof (proj1 (gb_parameter _ _ Hgb _ _ _) Hpar) as HparG.
    destruct (po_parameter _ Hpos _ _ _ HparG) as [_ Hne].
    rewrite (proj2 (ne_ac_usage t Hne)) in Hu.
    exact (has_source_parameter _ _ _ Hc t r b Hu HparG).
  - intros t f Hu Huse.
    pose proof (gb_use _ _ Hgb _ _ Huse) as HuseG.
    pose proof (po_use _ Hpos _ _ HuseG) as Hne.
    rewrite (proj2 (ne_ac_usage t Hne)) in Hu. apply cu_any_usage_old.
    exact (any_usage_use _ _ _ Hc t f Hu HuseG).
  - intros t f Hu Ha.
    destruct (cu_alias S t f Ha) as [Ha' | (_ & -> & _)]; [ | reflexivity ].
    destruct (node_eq_dec t Node_all_constants) as [-> | Hne];
      [ rewrite Hna in Ha'; discriminate Ha' | ].
    rewrite (proj1 (ne_ac_usage t Hne)) in Hu. apply cu_any_usage_old.
    exact (any_usage_alias _ _ _ Hc t f Hu Ha').
  - intros b r f Hu Hcons Hl.
    pose proof (proj1 (gb_constructor _ _ Hgb _ _ _) Hcons) as HconsG.
    pose proof (po_constructor _ Hpos _ _ _ HconsG) as Hne.
    rewrite (proj1 (ne_ac_usage b Hne)) in Hu. apply cu_any_usage_old.
    exact (any_usage_constructor _ _ _ Hc b r f Hu HconsG Hl).
  - intros b r f Has Harg. apply cu_any_usage_old.
    exact (any_usage_argument _ _ _ Hc b r f Has
             (proj1 (gb_argument _ _ Hgb _ _ _) Harg)).
  - intros x Hg.
    exact (any_source_input _ _ _ Hc x (gb_any_source _ _ Hgb x Hg)).
  - intros x Hg.
    exact (any_source_zero_alloc _ _ _ Hc x (gb_zero _ _ Hgb x Hg)).
  - intros i x Hu Hg.
    pose proof (gb_any_source_if_used _ _ Hgb _ _ Hg) as HgG.
    pose proof (po_any_source_if_used _ Hpos _ _ HgG) as Hne.
    rewrite (proj1 (ne_ac_usage i Hne)) in Hu.
    exact (any_source_if_used _ _ _ Hc i x Hu HgG).
  - intros f t Ha Has.
    destruct (cu_alias S t f Ha) as [Ha' | (_ & _ & Hast)];
      [ exact (any_source_alias _ _ _ Hc f t Ha' Has) | exact Hast ].
  - intros b r t Hu Hpar.
    pose proof (proj1 (gb_parameter _ _ Hgb _ _ _) Hpar) as HparG.
    destruct (po_parameter _ Hpos _ _ _ HparG) as [Hne _].
    rewrite (proj1 (ne_ac_usage b Hne)) in Hu.
    exact (any_source_parameter _ _ _ Hc b r t Hu HparG).
  - intros b r t Has Hacc Hl.
    exact (any_source_accessor _ _ _ Hc b r t Has
             (proj1 (gb_accessor _ _ Hgb _ _ _) Hacc) Hl).
  - intros f t Hs Huse.
    exact (any_source_use _ _ _ Hc f t Hs (gb_use _ _ Hgb _ _ Huse)).
  - intros x Hg. apply cu_any_usage_old.
    exact (any_usage_input _ _ _ Hc x (gb_any_usage _ _ Hgb x Hg)).
  - intros t r b Hacc Hau.
    exact (usages_accessor _ _ _ Hc t r b
             (proj1 (gb_accessor _ _ Hgb _ _ _) Hacc)
             (proj1 (cu_any_usage_false _ _ Hau))).
  - intros f r b Harg Hau.
    exact (usages_argument _ _ _ Hc f r b
             (proj1 (gb_argument _ _ Hgb _ _ _) Harg)
             (proj1 (cu_any_usage_false _ _ Hau))).
  - intros t f u Hus Hau Ha Hauf.
    destruct (cu_alias S t f Ha) as [Ha' | (_ & -> & _)];
      [ | cbn in Hauf; discriminate Hauf ].
    exact (usages_alias _ _ _ Hc t f u Hus
             (proj1 (cu_any_usage_false _ _ Hau)) Ha'
             (proj1 (cu_any_usage_false _ _ Hauf))).
  - intros f r b Hcons Has.
    exact (sources_constructor _ _ _ Hc f r b
             (proj1 (gb_constructor _ _ Hgb _ _ _) Hcons) Has).
  - intros t r b Hpar Has.
    exact (sources_parameter _ _ _ Hc t r b
             (proj1 (gb_parameter _ _ Hgb _ _ _) Hpar) Has).
  - intros f t s Hs Hasf Ha Hast.
    destruct (cu_alias S t f Ha) as [Ha' | (_ & _ & Hast')];
      [ exact (sources_alias _ _ _ Hc f t s Hs Hasf Ha' Hast)
      | cbn in Hast; rewrite Hast' in Hast; discriminate Hast ].
  - intros b r f Hu Hcons Hl.
    pose proof (proj1 (gb_constructor _ _ Hgb _ _ _) Hcons) as HconsG.
    pose proof (po_constructor _ Hpos _ _ _ HconsG) as Hne.
    rewrite (proj1 (ne_ac_usage b Hne)) in Hu.
    exact (escaping_field_rule _ _ _ Hc b r f Hu HconsG Hl).
  - intros b r t Has Hacc Hl.
    exact (reading_field_rule _ _ _ Hc b r t Has
             (proj1 (gb_accessor _ _ Hgb _ _ _) Hacc) Hl).
  - intros b r f Hcons Hu Hl.
    pose proof (proj1 (gb_constructor _ _ Hgb _ _ _) Hcons) as HconsG.
    pose proof (po_constructor _ Hpos _ _ _ HconsG) as Hne.
    rewrite (proj1 (ne_ac_usage b Hne)) in Hu.
    exact (field_used_any_usage _ _ _ Hc b r f HconsG Hu Hl).
  - intros b r f u v Hcons Hus Hacc Hu.
    pose proof (proj1 (gb_constructor _ _ Hgb _ _ _) Hcons) as HconsG.
    pose proof (proj1 (gb_accessor _ _ Hgb _ _ _) Hacc) as HaccG.
    pose proof (po_accessor _ Hpos _ _ _ HaccG) as Hne.
    rewrite (proj2 (ne_ac_usage v Hne)) in Hu.
    exact (field_used_usage _ _ _ Hc b r f u v HconsG Hus HaccG Hu).
  - intros b r f x Hcons Hu Hrd Hux.
    pose proof (proj1 (gb_constructor _ _ Hgb _ _ _) Hcons) as HconsG.
    pose proof (po_constructor _ Hpos _ _ _ HconsG) as Hne.
    rewrite (proj1 (ne_ac_usage b Hne)) in Hu.
    destruct (node_eq_dec x Node_all_constants) as [-> | Hnex];
      [ cbn in Hrd; rewrite Hnr in Hrd; discriminate Hrd | ].
    rewrite (proj2 (ne_ac_usage x Hnex)) in Hux.
    exact (field_used_reading _ _ _ Hc b r f x HconsG Hu Hrd Hux).
  - intros u b r f v Hcons Hsrc Hl Hu Hacc Huv.
    pose proof (proj1 (gb_constructor _ _ Hgb _ _ _) Hcons) as HconsG.
    pose proof (po_constructor _ Hpos _ _ _ HconsG) as Hneb.
    pose proof (proj1 (gb_accessor _ _ Hgb _ _ _) Hacc) as HaccG.
    pose proof (po_accessor _ Hpos _ _ _ HaccG) as Hnev.
    rewrite (proj1 (ne_ac_usage b Hneb)) in Hu.
    rewrite (proj2 (ne_ac_usage v Hnev)) in Huv.
    exact (field_used_local_source _ _ _ Hc u b r f v HconsG Hsrc Hl Hu
             HaccG Huv).
Qed.

Lemma consts_used_exact :
  field_used_exact is_local G S -> field_used_exact is_local G' S'.
Proof.
  intros Hex base r Hf. destruct (Hex base r Hf) as (f & Hcons & H).
  exists f. split; [ apply (gb_constructor _ _ Hgb); exact Hcons | ].
  destruct H as [[Hu Hl] | [(u & v & Hus & Hacc & Huv)
                 | [(x & Hu & Hrd & Hux) | (u & v & Hsrc & Hl & Hu & Hacc
                                            & Huv)]]].
  - left. split; [ apply cu_any_usage_old; exact Hu | exact Hl ].
  - right. left. exists u, v.
    split; [ exact Hus | split; [ apply (gb_accessor _ _ Hgb); exact Hacc
                                | apply cu_has_usage_old; exact Huv ] ].
  - right. right. left. exists x.
    split; [ apply cu_any_usage_old; exact Hu | ].
    split; [ exact Hrd | apply cu_has_usage_old; exact Hux ].
  - right. right. right. exists u, v.
    split; [ exact Hsrc | split; [ exact Hl | ] ].
    split; [ apply cu_any_usage_old; exact Hu | ].
    split; [ apply (gb_accessor _ _ Hgb); exact Hacc
           | apply cu_has_usage_old; exact Huv ].
Qed.

End Closed.

(* ======================================================================== *)
(* A least solution has nothing from the constants                          *)
(* ======================================================================== *)

Lemma least_no_consts (is_local : field -> bool) (F : facts) (T : tables) :
  least_solution is_local F T -> positions_ok (graph_of_facts F) ->
  (∀ n, s_alias (solution_of_tables T) Node_all_constants n = false)
  ∧ (∀ r,
       s_reading_field (solution_of_tables T) r Node_all_constants = false).
Proof.
  intros Hleast Hpos.
  assert (Hr : ∀ r, s_reading_field (solution_of_tables T) r
                      Node_all_constants = false).
  { intros r.
    destruct (s_reading_field (solution_of_tables T) r Node_all_constants)
      eqn:E; [ exfalso | reflexivity ].
    pose proof (FlambdaReaperLeastProofs.least_derivable is_local F T Hleast
                  (Atom_reading_field r Node_all_constants) E) as D.
    inversion D; subst.
    exact (po_accessor _ Hpos _ _ _ ltac:(eassumption) eq_refl). }
  split; [ | exact Hr ].
  intros n. destruct (s_alias (solution_of_tables T) Node_all_constants n)
    eqn:E; [ exfalso | reflexivity ].
  pose proof (FlambdaReaperLeastProofs.least_derivable is_local F T Hleast
                (Atom_alias Node_all_constants n) E) as D.
  inversion D; subst;
    try (exact (po_alias _ Hpos _ _ ltac:(eassumption) eq_refl));
    try (exact (proj2 (po_propagate _ Hpos _ _ _ ltac:(eassumption))
                  eq_refl));
    try (exact (po_alias_if _ Hpos _ _ _ ltac:(eassumption) eq_refl));
    try (exact (po_accessor _ Hpos _ _ _ ltac:(eassumption) eq_refl));
    try (exact (proj2 (po_parameter _ Hpos _ _ _ ltac:(eassumption))
                  eq_refl)).
  match goal with
  | H : derivable _ _ _ (Atom_reading_field ?r Node_all_constants) |- _ =>
    pose proof (FlambdaReaperLeastProofs.least_holds is_local F T Hleast _
                  H) as Hh; cbn [holds] in Hh; rewrite Hr in Hh;
    discriminate Hh
  end.
Qed.

(* ======================================================================== *)
(* The continuations of the erased term                                     *)
(* ======================================================================== *)

Lemma continuations_erase :
  (∀ e : expr, ∀ c, continuations_of (erase c e) = continuations_of e)
  ∧ (∀ h : continuation_handler, ∀ c,
       continuations_in_handler (erase_handler c h)
       = continuations_in_handler h).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - intros x kind d body IH c.
    destruct d as [s | p args | | | | | ]; cbn [erase];
      try (cbn [continuations_of]; apply IH).
    destruct p as [ | | | | r]; try (cbn [continuations_of]; apply IH).
    destruct (erase_region_primitive x r args);
      cbn [continuations_of]; apply IH.
  - intros bindings body Hcodes IH c. cbn [erase continuations_of].
    rewrite IH. f_equal. rewrite flat_map_concat_map, map_map,
      flat_map_concat_map. f_equal. apply map_ext_in.
    intros b Hin. destruct b as [id cd | | ]; try reflexivity.
    destruct cd as [params mc ret exn md body'].
    cbn [erase_binding erase_code continuations_in_binding
           continuations_in_code].
    do 2 f_equal. exact (Hcodes id _ Hin []).
  - intros k h body IHh IH c. cbn [erase continuations_of].
    rewrite IHh, IH. reflexivity.
  - intros handlers body IHh IH c. cbn [erase continuations_of].
    rewrite IH. f_equal. rewrite flat_map_concat_map, map_map,
      flat_map_concat_map. f_equal. apply map_ext_in.
    intros [k h] Hin. cbn [fst snd]. rewrite (IHh k h Hin _). reflexivity.
  - reflexivity.
  - intros ac c. cbn [erase continuations_of].
    apply apply_cont_continuations_erase.
  - intros sc arms c. cbn [erase continuations_of].
    rewrite flat_map_concat_map, map_map, flat_map_concat_map. f_equal.
    apply map_ext. intros [i ac]. apply apply_cont_continuations_erase.
  - reflexivity.
  - intros params md hbody IH c. cbn [erase_handler
      continuations_in_handler]. apply IH.
Qed.

Lemma fresh_name_erase (e : expr) :
  among (fresh_name (names_of e)) (continuations_of (erase [] e)) = false.
Proof.
  rewrite (proj1 continuations_erase e []).
  apply not_true_iff_false. intros E. apply among_In in E.
  pose proof (FlambdaNameAbstractionProofs.fresh_name_fresh (names_of e))
    as Hff.
  apply not_true_iff_false in Hff. apply Hff. apply among_In.
  unfold names_of. apply in_or_app. right. exact E.
Qed.

(* ======================================================================== *)
(* The commutation                                                          *)
(* ======================================================================== *)

Lemma parameters_not_consts (l : list fact) :
  (∀ fct, fct ∈ l -> pos_ok fct) ->
  ∀ (x : node) (i : nat) (y : node),
    (x, i, y) ∈ f_parameter (facts_of_list l) -> y <> Node_all_constants.
Proof.
  intros H x i y Hin. pose proof (graph_positions l H) as Hpos.
  exact (proj2 (po_parameter _ Hpos x i y Hin)).
Qed.

Lemma rebuild_erase_commutes_with (flag : reaper_preserve_direct_calls)
    (u : unit_interface) (e : expr) (sol sol' : solution) (b : bool) :
  regions_plain [] e = true ->
  (∀ n : node, n <> Node_all_constants ->
     s_has_usage sol' n = s_has_usage sol n) ->
  s_has_source sol' = s_has_source sol ->
  s_any_source sol' = s_any_source sol ->
  s_sources sol' = s_sources sol ->
  s_field_used sol' = s_field_used sol ->
  erase [] (rebuild_expr (traverse_unit flag u e) sol b
              (static_names code_ids_bound e) (fresh_name (names_of e)) None
              FlambdaReaperRebuild.empty_env e)
  = rebuild_expr (traverse_unit flag u (erase [] e)) sol' b
      (static_names code_ids_bound (erase [] e)) (fresh_name (names_of e))
      None FlambdaReaperRebuild.empty_env (erase [] e).
Proof.
  intros Hpl Hu Hsrc Has Hss Hfu.
  destruct (traverse_unit_rel flag u e Hpl) as (l & l' & HF & HF' & Hrel).
  destruct (traverse_unit_positions flag u e) as (l0 & HF0 & Hpos0).
  destruct Hrel as [(HC & HA & HP & _) _].
  rewrite (proj1 (static_names_erase code_ids_bound
                    ltac:(intros [ | | ]; reflexivity)) e []).
  assert (Hpac : ∀ (x : node) (i : nat) (y : node),
            (x, i, y) ∈ f_parameter (traverse_unit flag u e) ->
            y <> Node_all_constants)
    by (rewrite HF0; apply parameters_not_consts, Hpos0).
  assert (HC' : f_constructor (traverse_unit flag u (erase [] e))
                = f_constructor (traverse_unit flag u e))
    by (rewrite HF, HF'; exact HC).
  assert (HA' : f_accessor (traverse_unit flag u (erase [] e))
                = f_accessor (traverse_unit flag u e))
    by (rewrite HF, HF'; exact HA).
  assert (HP' : f_parameter (traverse_unit flag u (erase [] e))
                = f_parameter (traverse_unit flag u e))
    by (rewrite HF, HF'; exact HP).
  refine (proj1 (rebuild_erase_commutes_gen _ _ _ _ _ _ _ HC' HA' HP' Hpac
                   Hu Hsrc Has Hss Hfu) e FlambdaReaperRebuild.empty_env
            FlambdaReaperRebuild.empty_env Hpl (env_rel_refl _) _).
  apply pending_ok_nil. reflexivity.
Qed.

Theorem rebuild_erase_commutes : rebuild_erase_commutes_statement.
Proof.
  intros flag u e T b Hpl.
  exact (rebuild_erase_commutes_with flag u e _ _ b Hpl
           (fun _ _ => eq_refl) eq_refl eq_refl eq_refl eq_refl).
Qed.

(* ======================================================================== *)
(* The refinement on the machine of regions                                 *)
(* ======================================================================== *)

Lemma erase_statics_empty : erase_statics empty_static_env = empty_static_env.
Proof. reflexivity. Qed.

(* From the refinement of the two terms with their regions dropped,
   that of the two terms on the machine of regions, if the target passes
   the check. *)
Lemma regions_refine_of_erased (cert : certificate) (base : symbol)
    (e r : expr) :
  region_names_fresh base e empty_static_env = true ->
  erasable e empty_static_env = true ->
  region_names_fresh base r empty_static_env = true ->
  erasable r empty_static_env = true ->
  region_check cert empty_static_env r = true ->
  refines_observably (erased_initial_state e empty_static_env)
    (erased_initial_state r empty_static_env) ->
  regions_refine_observably empty_static_env base e r.
Proof.
  intros Hfe Hee Hfr Her Hcheck [H1 [H2 H3]].
  pose proof (FlambdaRegionsCheckSoundProofs.region_check_sound cert
                empty_static_env base r Hcheck Hfr Her) as Hsafe.
  split; [ | split ].
  - intros fuel k vs rs Hrun.
    destruct (FlambdaRegionsCallsProofs.erasure_exits_erased e
                empty_static_env base fuel k vs rs Hfe Hee Hrun)
      as (us & hp & Hb & Hvs).
    destruct (H1 k us hp Hb) as (us' & Hb' & Hobs).
    destruct (FlambdaRegionsCallsProofs.erasure_adds_no_exit_erased r
                empty_static_env base k us' hp Hfr Her Hb')
      as [fuel' [(fault & Hf) | (ws & rs' & Hr & Hws)]].
    + exfalso. pose proof (Hsafe fuel' fault Hf) as Hw.
      pose proof (FlambdaSemanticsProofs.behaviour_unique _ _ _ Hb' Hw)
        as E.
      discriminate E.
    + exists fuel', ws, rs', us, us'. repeat split; assumption.
  - intros fuel' k ws rs' Hrun.
    destruct (FlambdaRegionsCallsProofs.erasure_exits_erased r
                empty_static_env base fuel' k ws rs' Hfr Her Hrun)
      as (us' & hp & Hb' & Hws).
    destruct (H2 k us' hp Hb') as [(us & Hb & Hobs) | Hw];
      [ | right; right; exact Hw ].
    destruct (FlambdaRegionsCallsProofs.erasure_adds_no_exit_erased e
                empty_static_env base k us hp Hfe Hee Hb)
      as [fuel [(fault & Hf) | (vs & rs & He & Hvs)]].
    + right. left. exists fuel, fault. exact Hf.
    + left. exists fuel, vs, rs, us, us'. repeat split; assumption.
  - intros fuel' fault Hf. apply H3. exact (Hsafe fuel' fault Hf).
Qed.

(* The refinement up to observation of the plain rebuild, for any
   solution closed under the rules, of which the fix of local fields and
   the post-processing hold, given as propositions, and any first fresh
   name that no continuation of the unit is
   (FlambdaReaperRebuildProofs.rebuild_with_refines_gen). *)
Lemma rebuild_refines_observably_gen (flag : reaper_preserve_direct_calls)
    (is_local : field -> bool) (u : unit_interface) (e : expr)
    (sol : solution) (b : bool) (ff : positive) :
  is_local_slots_only is_local ->
  in_flow_fragment e = true ->
  any_kind_arguments e = true ->
  results_kinded [] [] e = true ->
  well_formed_unit u e = true ->
  exits_one_value u e = true ->
  closed is_local (graph_of_facts (traverse_unit flag u e)) sol ->
  local_reads_used is_local (graph_of_facts (traverse_unit flag u e)) sol ->
  field_used_exact is_local (graph_of_facts (traverse_unit flag u e)) sol ->
  among ff (continuations_of e) = false ->
  refines_observably (State e [] [] empty_static_env empty_heap no_reads)
    (State (rebuild_expr (traverse_unit flag u e) sol b
              (static_names code_ids_bound e) ff None
              FlambdaReaperRebuild.empty_env e)
       [] [] empty_static_env empty_heap no_reads).
Proof.
  intros Hlocal Hfrag Hdc Hrk Hwf Hexits Hc Hlrs Hexact Hfresh.
  assert (Hkinded : FlambdaReaperRebuildProofs.calls_kinded [] e = true).
  { apply (proj1 FlambdaReaperRebuildProofs.calls_kinded_of_results e [] []
             [] Hrk).
    intros k l Hl. discriminate Hl. }
  pose proof (proj1 (FlambdaReaperRebuildProofs.any_kind_arguments_checked
                       (unit_codes e)) e Hdc) as Hcf.
  exact (proj1 (FlambdaReaperRebuildProofs.exits_rel_flow false _ _
                  (FlambdaReaperRebuildProofs.rebuild_with_refines_gen false
                     false flag is_local u e sol b None ff Hlocal Hfrag Hcf
                     Hkinded Hwf Hexits Hc Hlrs Hexact Hfresh
                     (fun _ _ _ _ _ _ _ _ _ _ _ _ => I)
                     (FlambdaReaperRebuildProofs.no_types_deps _)
                     (FlambdaReaperRebuildProofs.no_types_conts e)
                     (fun _ => Hrk)))).
Qed.

Theorem least_rebuild_regions_refines :
  least_rebuild_regions_refines_statement.
Proof.
  intros flag is_local u e T b cert base Hpl Hslots Hlocal Hfrag Hdc Hrk
    Hwf Hexits Hleast Hfe Hee Hfr Her Hcheck.
  apply (regions_refine_of_erased cert); try assumption.
  unfold erased_initial_state. rewrite erase_statics_empty.
  unfold rebuild_unit_from_tables, rebuild_unit.
  rewrite (rebuild_erase_commutes_with flag u e (solution_of_tables T)
             (consts_used (solution_of_tables T)) b Hpl
             (fun n Hn => proj2 (cu_usage (solution_of_tables T) n Hn))
             eq_refl eq_refl eq_refl eq_refl).
  remember (traverse_unit flag u e) as F eqn:HFdef.
  remember (traverse_unit flag u (erase [] e)) as F' eqn:HF'def.
  remember (solution_of_tables T) as sol eqn:Hsoldef.
  destruct (traverse_unit_rel flag u e Hpl) as (l & l' & HF & HF' & Hrel).
  destruct (traverse_unit_positions flag u e) as (l0 & HF0 & Hpos0).
  assert (Hpos : positions_ok (graph_of_facts F))
    by (rewrite HFdef, HF0; apply graph_positions, Hpos0).
  assert (Hgb : graph_below (graph_of_facts F) (graph_of_facts F'))
    by (rewrite HFdef, HF'def, HF, HF'; apply graph_below_of_rel, Hrel).
  assert (Hc : closed is_local (graph_of_facts F) sol)
    by (rewrite Hsoldef;
        exact (FlambdaReaperSolutionProofs.check_closed_sound is_local F T
                 (proj1 Hleast))).
  assert (Hexact : field_used_exact is_local (graph_of_facts F) sol)
    by (rewrite Hsoldef;
        exact (FlambdaReaperSolutionProofs.check_field_used_exact_sound
                 is_local F T
                 (FlambdaReaperLeastProofs.least_field_used_exact is_local F T
                    Hleast))).
  destruct (least_no_consts is_local F T Hleast Hpos) as [Hna Hnr].
  rewrite <- Hsoldef in Hna, Hnr.
  assert (Hc' : closed is_local (graph_of_facts F') (consts_used sol))
    by exact (consts_used_closed is_local _ _ sol Hc Hna Hnr Hpos Hgb).
  assert (Hexact' : field_used_exact is_local (graph_of_facts F')
                      (consts_used sol))
    by exact (consts_used_exact is_local _ _ sol Hgb Hexact).
  assert (Hlrs' : local_reads_used is_local (graph_of_facts F')
                    (consts_used sol)).
  { intros base' rel to Has Hacc Hl. exfalso.
    destruct Hlocal as [Hno | Hpf]; [ rewrite Hno in Hl; discriminate Hl | ].
    rewrite HF'def in Hacc.
    destruct (FlambdaReaperFlowProofs.traverse_unit_reads_no_slot flag u
                (erase [] e) Hpf to rel base' Hacc) as [Hw Hfs].
    destruct (Hslots rel Hl) as [(w & ->) | (fs & ->)];
      [ exact (Hw w eq_refl) | exact (Hfs fs eq_refl) ]. }
  rewrite HF'def in Hc', Hlrs', Hexact'.
  pose proof (rebuild_refines_observably_gen flag is_local u (erase [] e)
                (consts_used sol) b (fresh_name (names_of e)) Hslots Hfrag
                Hdc Hrk Hwf Hexits Hc' Hlrs' Hexact' (fresh_name_erase e))
    as Hbase.
  rewrite <- HF'def in Hbase. exact Hbase.
Qed.

(* ======================================================================== *)
(* What the rebuild keeps                                                   *)
(* ======================================================================== *)

Theorem rebuild_keeps_erasable : rebuild_keeps_erasable_statement.
Proof.
  intros F T b e.
  exact (FlambdaReaperRegionsKeepsProofs.rebuild_keeps_erasable F T b e).
Qed.

Theorem rebuild_keeps_fresh : rebuild_keeps_fresh_statement.
Proof.
  intros F T b e base.
  exact (FlambdaReaperRegionsKeepsProofs.rebuild_keeps_fresh F T b e base).
Qed.

Theorem least_rebuild_regions_refines_checked :
  least_rebuild_regions_refines_checked_statement.
Proof.
  intros flag is_local u e T b cert base Hpl Hslots Hlocal Hfrag Hdc Hrk
    Hwf Hexits Hleast Hfe Hee Hcheck.
  apply (least_rebuild_regions_refines flag is_local u e T b cert base);
    try assumption.
  - exact (rebuild_keeps_fresh _ T b e base Hpl Hfe).
  - exact (rebuild_keeps_erasable _ T b e Hpl Hee).
Qed.

(* ======================================================================== *)
(* Examples                                                                 *)
(* ======================================================================== *)

Import FlambdaReaperRegionsSpec.RegionsExamples.

(* The state that a run is at, or exits in. *)
Definition run_state (r : region_step_result) : region_state :=
  match r with
  | Region_exit _ _ rs | Region_step rs => rs
  | Region_undefined _ => initial_region_state Invalid empty_static_env xH
  end.

Ltac ex_run :=
  match goal with
  | |- ∃ rs : region_state, ?run = _ => exists (run_state run)
  end;
  vm_compute; reflexivity.

Lemma no_local_slots : is_local_slots_only (fun _ => false).
Proof. intros f H. discriminate H. Qed.

Theorem regions_example : regions_example_statement.
Proof.
  assert (Hleast : least_solution (fun _ => false)
                     (traverse_unit Auto interface two_blocks)
                     two_blocks_tables)
    by (apply (FlambdaReaperLeastProofs.check_least_sound _ _ 100);
        vm_compute; reflexivity).
  assert (Hreb : rebuild_unit_from_tables
                   (traverse_unit Auto interface two_blocks)
                   two_blocks_tables true two_blocks
                 = two_blocks_rebuilt)
    by (vm_compute; reflexivity).
  split; [ vm_compute; reflexivity | ]. split; [ exact Hleast | ].
  do 7 (split; [ vm_compute; reflexivity | ]).
  split; [ exact Hreb | ].
  do 3 (split; [ vm_compute; reflexivity | ]).
  split; [ ex_run | ]. split; [ ex_run | ].
  rewrite <- Hreb.
  apply (least_rebuild_regions_refines Auto (fun _ => false) interface
           two_blocks two_blocks_tables true cert base);
    try (vm_compute; reflexivity); try (rewrite Hreb; vm_compute; reflexivity).
  - exact no_local_slots.
  - left. intros f. reflexivity.
  - exact Hleast.
Qed.

Theorem boxed_poison_counterexample :
  boxed_poison_counterexample_statement.
Proof.
  assert (Hrun : region_run 10
                   (initial_region_state boxed_rebuilt empty_static_env base)
                 = Region_undefined Not_of_the_conversion)
    by (vm_compute; reflexivity).
  assert (Hfr : region_names_fresh base boxed_rebuilt empty_static_env = true)
    by (vm_compute; reflexivity).
  split; [ vm_compute; reflexivity | ].
  split; [ apply (FlambdaReaperLeastProofs.check_least_sound _ _ 100);
           vm_compute; reflexivity | ].
  do 7 (split; [ vm_compute; reflexivity | ]).
  split; [ vm_compute; reflexivity | ].
  split; [ exact Hfr | ]. split; [ vm_compute; reflexivity | ].
  split.
  { intros c. destruct (region_check c empty_static_env boxed_rebuilt)
      eqn:E; [ exfalso | reflexivity ].
    pose proof (FlambdaRegionsCheckProofs.region_check_fault c
                  empty_static_env base boxed_rebuilt 10 Not_of_the_conversion
                  E Hfr Hrun) as H.
    discriminate H. }
  split; [ ex_run | ]. split; [ exact Hrun | ].
  intros (_ & _ & H3).
  pose proof (H3 10 Not_of_the_conversion Hrun) as Hw.
  assert (He : behaves (erased_initial_state boxed empty_static_env)
                 (Exits 1%positive [Value_const (Tagged_immediate 0)] []))
    by (exists 10, []; vm_compute; reflexivity).
  pose proof (FlambdaSemanticsProofs.behaviour_unique _ _ _ He Hw) as E.
  discriminate E.
Qed.

Theorem region_check_not_kept : region_check_not_kept_statement.
Proof.
  destruct boxed_poison_counterexample
    as (Hpl & Hleast & _ & _ & _ & _ & _ & _ & _ & Hreb & _ & _ & Hc & _).
  split; [ exact Hpl | ]. split; [ exact Hleast | ].
  split; [ vm_compute; reflexivity | ].
  rewrite Hreb. exact Hc.
Qed.

Theorem local_set_counterexample : local_set_counterexample_statement.
Proof.
  split; [ vm_compute; reflexivity | ].
  split; [ apply (FlambdaReaperLeastProofs.check_least_sound _ _ 100);
           vm_compute; reflexivity | ].
  do 4 (split; [ vm_compute; reflexivity | ]).
  split.
  { intros H. apply (f_equal (fun e => length (names_of e))) in H.
    vm_compute in H. discriminate H. }
  split; ex_run.
Qed.

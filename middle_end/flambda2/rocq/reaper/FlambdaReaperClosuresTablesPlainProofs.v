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

(* The proofs of FlambdaReaperClosuresTablesPlainSpec.v. Nothing here
   needs review.

   [alias_into_l], [any_source_into_l], [sources_into_l]: what the rules
   give a node only read into, local fields included (the lemmas of
   FlambdaReaperLeastProofs.v are of a field that is not local).
   [read_dominated]: a read of the slot off a node [u] of no any source
   whose sources with a source hold the simple [n] alone is dominated by
   the allocation [a0] that dominates [n]. [case_read]: so is each read
   of [field_used_as] of a closure of the set, by the rules of
   [cannot_change_representation0]. [not_unboxed_absurd]: the field is
   then unboxed. *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperFlowSpec FlambdaReaperLeastSpec FlambdaReaperRebuild
  FlambdaReaperCallingConventions FlambdaReaperUnboxing
  FlambdaReaperUnboxingRefinesSpec FlambdaReaperUnboxingRefinesCallsSpec
  FlambdaReaperUnboxingRefinesClosuresSpec FlambdaReaperClosuresTablesSpec
  FlambdaReaperClosuresTablesFieldsSpec FlambdaReaperClosuresTablesSlotsSpec
  FlambdaReaperTraversalSlotsSpec FlambdaReaperClosuresTablesPlainSpec.
From OxCamlFlambda2 Require FlambdaReaperFlowProofs FlambdaReaperTypesProofs
  FlambdaReaperTypesTraversalProofs FlambdaReaperUnboxingProofs
  FlambdaReaperUnboxingSpec FlambdaReaperUnboxingRefinesProofs
  FlambdaReaperClosuresTablesProofs FlambdaReaperClosuresTablesFieldsProofs
  FlambdaReaperClosuresTablesSlotsProofs FlambdaReaperLeastProofs
  FlambdaReaperTraversalSlotsProofs FlambdaReaperUnboxingFuelSpec
  FlambdaReaperTypes.
From OxCamlFlambda2 Require Import FlambdaReaperSolutionProofs.
Import ListNotations.

Module TP := FlambdaReaperTypesProofs.
Module TTP := FlambdaReaperTypesTraversalProofs.
Module URP := FlambdaReaperUnboxingRefinesProofs.
Module CTP := FlambdaReaperClosuresTablesProofs.
Module FTP := FlambdaReaperClosuresTablesFieldsProofs.
Module STP := FlambdaReaperClosuresTablesSlotsProofs.
Module LP := FlambdaReaperLeastProofs.
Module TSP := FlambdaReaperTraversalSlotsProofs.
Module US := FlambdaReaperUnboxingSpec.
Module UP := FlambdaReaperUnboxingProofs.
Module UF := FlambdaReaperUnboxingFuelSpec.
Module Ty := FlambdaReaperTypes.

Ltac unfold_rels :=
  cbn [graph_of_facts solution_of_tables holds g_alias g_use g_accessor
       g_constructor g_argument g_parameter g_propagate
       g_alias_if_any_source g_any_usage g_any_source g_any_source_if_used
       g_zero_alloc_source s_alias s_any_usage s_any_source s_has_usage
       s_has_source s_usages s_sources s_reading_field s_escaping_field
       s_field_used] in *.

(* ======================================================================== *)
(* A node only read into                                                    *)
(* ======================================================================== *)

Section Read_only.

Variable is_local : field -> bool.
Variable F : facts.
Variable T : tables.
Variable y : node.
Variable f : field.
Variable base : node.
Hypothesis Hro : LP.read_only F y f base false.

Local Abbreviation G := (graph_of_facts F).
Local Abbreviation S := (solution_of_tables T).
Local Abbreviation D := (derivable is_local G S).

Ltac ro_absurd :=
  exfalso;
  first
    [ eapply (LP.ro_alias _ _ _ _ _ Hro); eassumption
    | eapply (LP.ro_propagate _ _ _ _ _ Hro); eassumption
    | eapply (LP.ro_parameter_to _ _ _ _ _ Hro); eassumption
    | eapply (LP.ro_parameter_base _ _ _ _ _ Hro); eassumption
    | eapply (LP.ro_constructor _ _ _ _ _ Hro); eassumption
    | eapply (LP.ro_any_source _ _ _ _ _ Hro); eassumption
    | eapply (LP.ro_zero_alloc _ _ _ _ _ Hro); eassumption
    | eapply (LP.ro_if_used _ _ _ _ _ Hro); eassumption
    | eapply (LP.ro_use _ _ _ _ _ Hro); eassumption ].

Ltac ro_read :=
  match goal with
  | H : (y, _, _) ∈ f_accessor F |- _ =>
    destruct (LP.ro_accessor _ _ _ _ _ Hro _ _ H) as [-> ->]
  end.

(* An alias of [y] is held at [f] by a node that [base] is a use of, or
   by a source of [base], or, where [base] has any source, by a node of
   any usage at a local field. *)
Lemma alias_into_l (from : node) :
  D (Atom_alias y from) ->
  (∃ b : node, g_constructor G b f from ∧ D (Atom_usages b base))
  ∨ (∃ bs : node, D (Atom_sources base bs) ∧ g_constructor G bs f from)
  ∨ (D (Atom_any_source base)
     ∧ ∃ b : node, D (Atom_any_usage b) ∧ g_constructor G b f from).
Proof.
  intros Hd. inversion Hd; subst; unfold_rels.
  - ro_absurd.
  - ro_absurd.
  - match goal with
    | H : (_, y, _) ∈ f_alias_if_any_source F |- _ =>
        destruct (LP.ro_if_any_source _ _ _ _ _ Hro _ _ H) as [E _];
        discriminate E
    end.
  - ro_read. left. eexists. eauto.
  - ro_read. left. eexists. eauto.
  - ro_absurd.
  - ro_read. right; left. eexists. eauto.
  - ro_read. right; left. eexists. eauto.
  - match goal with
    | H : derivable _ _ _ (Atom_reading_field _ _) |- _ =>
        inversion H; subst; unfold_rels
    end.
    ro_read.
    match goal with
    | H : derivable _ _ _ (Atom_escaping_field _ _) |- _ =>
        inversion H; subst; unfold_rels
    end.
    right; right. split; [ assumption | ]. eexists. eauto.
  - ro_absurd.
Qed.

(* [y] has any source through an alias, or as [base] has. *)
Lemma any_source_into_l :
  D (Atom_any_source y) ->
  (∃ from : node, D (Atom_alias y from) ∧ D (Atom_any_source from))
  ∨ D (Atom_any_source base).
Proof.
  intros Hd. inversion Hd; subst; unfold_rels.
  - ro_absurd.
  - ro_absurd.
  - ro_absurd.
  - left. eauto.
  - ro_absurd.
  - ro_read. right. assumption.
  - ro_absurd.
Qed.

(* [y] has the sources of what it is an alias of. *)
Lemma sources_into_l (s : node) :
  D (Atom_sources y s) ->
  ∃ from : node, D (Atom_sources from s) ∧ D (Atom_alias y from).
Proof.
  intros Hd. inversion Hd; subst; unfold_rels.
  - ro_absurd.
  - ro_absurd.
  - eexists. eauto.
Qed.

End Read_only.

(* ======================================================================== *)
(* Lists                                                                    *)
(* ======================================================================== *)

Lemma dedup_single (l : list node) (s : node) :
  s ∈ l -> (∀ z, z ∈ l -> z = s) -> Keyed.dedup node_eqb l = [s].
Proof.
  destruct l as [ | z rest]; [ intros [] | ].
  intros _ Hall. cbn [Keyed.dedup].
  rewrite (Hall z (or_introl eq_refl)). f_equal.
  assert (H : ∀ k, k ∈ Keyed.dedup node_eqb rest -> k = s)
    by (intros k Hk; apply Hall; right; exact (URP.dedup_in _ _ Hk)).
  unfold Keyed.without. revert H.
  generalize (Keyed.dedup node_eqb rest) as l'.
  induction l' as [ | k ks IH]; intros H; [ reflexivity | ].
  cbn [filter]. rewrite (H k (or_introl eq_refl)).
  unfold Keyed.among. cbn [existsb]. rewrite node_eqb_refl. cbn [negb orb].
  apply IH. intros k' Hk'. apply H. right. exact Hk'.
Qed.

(* A field that [mk_unboxed_fields] keeps is made by its step. *)
Lemma mk_step_in (F : facts) (T : tables) (cm : finite_map variable)
    (dominated : list (node * node)) (to_unbox : list node) (fuel : nat)
    (block : node) :
  ∀ (fields : list (field * field_use)) (next : positive) (f : field)
    (u : unboxed_field),
    (f, u) ∈ fst (mk_unboxed_fields F T cm dominated to_unbox fuel block
                    fields next) ->
    ∃ (use : field_use) (n1 n2 : positive),
      (f, use) ∈ fields
      ∧ URP.field_step F T cm dominated to_unbox fuel block f use n1
        = (Some u, n2).
Proof.
  induction fields as [ | [f' use] fields IH]; intros next f u Hin.
  - rewrite URP.mk_unboxed_fields_nil in Hin. destruct Hin.
  - rewrite URP.mk_unboxed_fields_cons in Hin.
    destruct (URP.field_step F T cm dominated to_unbox fuel block f' use next)
      as [u0 n0] eqn:Hs.
    destruct (mk_unboxed_fields F T cm dominated to_unbox fuel block fields
                n0)
      as [rest n1] eqn:Hr.
    cbn [fst] in Hin.
    assert (Hrest : (f, u) ∈ rest ->
                    ∃ (use' : field_use) (m1 m2 : positive),
                      (f, use') ∈ (f', use) :: fields
                      ∧ URP.field_step F T cm dominated to_unbox fuel block
                          f use' m1
                        = (Some u, m2)).
    { intros H. destruct (IH n0 f u) as (use' & m1 & m2 & Hu & Hst);
        [ rewrite Hr; exact H | ].
      exists use', m1, m2. split; [ right; exact Hu | exact Hst ]. }
    destruct u0 as [u0 | ]; [ | exact (Hrest Hin) ].
    destruct Hin as [[= <- <-] | Hin]; [ | exact (Hrest Hin) ].
    exists use, next, n0. split; [ left; reflexivity | exact Hs ].
Qed.

(* ======================================================================== *)
(* The field of a set                                                       *)
(* ======================================================================== *)

Section Plain.

Variable is_local : field -> bool.
Variable F : facts.
Variable T : tables.
Hypothesis Hleast : least_solution is_local F T.

Local Abbreviation G := (graph_of_facts F).
Local Abbreviation S := (solution_of_tables T).
Local Abbreviation D := (derivable is_local G S).
Local Abbreviation ctx := (Ty.plain_context is_local F T []).

Variables crr0 crr1 crr cu0 cu : node -> bool.
Variable cannot_change : list node.
Variable max_unbox_size : nat.
Variable lambda_lifting : bool.
Hypothesis Hclosed :
  US.unboxing_closed is_local G S (fun n => mem_node n cannot_change)
    max_unbox_size lambda_lifting crr0 crr1 crr cu0 cu.
Variable to_unbox : list node.
Hypothesis Hcu : ∀ x, mem_node x to_unbox = true -> cu x = false.
Variable dominated : list (node * node).
Hypothesis Hdom : check_dominated T dominated = true.

(* The allocation [x0] of the set, which can change its representation,
   and its closures, which hold [x0] in a function slot and [n] alone at
   the value slot [w]. *)
Variable x0 : node.
Hypothesis Hcrr : crr x0 = false.
Variable w : value_slot.
Variable n : node.
Hypothesis Hx0 : ∀ b, g_constructor G x0 (Field_value_slot w) b <-> b = n.
Hypothesis Hcl :
  ∀ c, c ∈ map snd (set_function_slots F x0) ->
    (∀ b, g_constructor G c (Field_value_slot w) b <-> b = n)
    ∧ ∃ g, g_constructor G c (Field_function_slot g) x0.
(* The reads of the slot are only read into. *)
Hypothesis Hread :
  ∀ y b, g_accessor G y (Field_value_slot w) b ->
    LP.read_only F y (Field_value_slot w) b false.

(* [n] is dominated by the allocation [a0] to unbox. *)
Variable a0 : node.
Hypothesis Ha0 : (a0, n) ∈ dominated.
Hypothesis Ha0u : mem_node a0 to_unbox = true.

Lemma closed_S : closed is_local G S.
Proof. exact (LP.least_closed is_local F T Hleast). Qed.

Lemma n_dominated : US.dominated_by_allocation_point S a0 n.
Proof.
  apply (proj1 (UP.check_dominated_sound T dominated Hdom a0 n)).
  apply mem_pair_In. exact Ha0.
Qed.

Lemma n_no_any_source : s_any_source S n = false.
Proof.
  destruct n_dominated as (_ & _ & Hnm).
  destruct (s_any_source S n) eqn:E; [ | reflexivity ].
  exfalso. apply Hnm. left. exact E.
Qed.

Lemma a0_no_any_usage : s_any_usage S a0 = false.
Proof.
  destruct (s_any_usage S a0) eqn:E; [ | reflexivity ].
  pose proof (US.cu_cu0 _ _ _ _ _ _ _ _ _ _ _ Hclosed a0
                (US.cu0_any_usage _ _ _ _ _ _ _ _ _ _ _ Hclosed a0 E))
    as H.
  rewrite (Hcu a0 Ha0u) in H. discriminate H.
Qed.

(* No closure of the set is of [cannot_change_representation0]: the set
   would not change its representation. *)
Lemma closure_not_crr0 (c : node) :
  c ∈ map snd (set_function_slots F x0) -> crr0 c = false.
Proof.
  intros Hc. destruct (Hcl c Hc) as [_ (g & Hg)].
  destruct (crr0 c) eqn:E; [ exfalso | reflexivity ].
  pose proof (US.crr1_function_slot _ _ _ _ _ _ _ _ _ _ _ Hclosed c _ x0 Hg
                eq_refl E)
    as H1.
  pose proof (US.crr_crr1 _ _ _ _ _ _ _ _ _ _ _ Hclosed x0 H1) as H2.
  rewrite Hcrr in H2. discriminate H2.
Qed.

Lemma sources_has_source (x s : node) :
  D (Atom_sources x s) -> s_has_source S s = true -> s_has_source S x = true.
Proof.
  pose proof closed_S as Hc.
  intros Hd. remember (Atom_sources x s) as at0 eqn:E. revert x s E.
  induction Hd; intros x' s' E Hs; try discriminate E; injection E as <- <-.
  - exact Hs.
  - exact Hs.
  - apply (LP.least_holds is_local F T Hleast) in Hd2. unfold_rels.
    exact (has_source_alias _ _ _ Hc from to_
             (IHHd1 from source eq_refl Hs) Hd2).
Qed.

(* A node that holds [n] alone at [w]. *)
Definition holds_n (s : node) : Prop :=
  ∀ b, g_constructor G s (Field_value_slot w) b -> b = n.

(* What a read [y] of [w] off a node [u] of no any source, whose sources
   with a source hold [n] alone at [w], is an alias of: [n], or a node
   with no source. *)
Lemma read_aliases (y u : node) :
  LP.read_only F y (Field_value_slot w) u false ->
  s_any_source S u = false ->
  (∀ s, s_sources S u s = true -> s_has_source S s = true -> holds_n s) ->
  ∀ from, D (Atom_alias y from) -> from = n ∨ s_has_source S from = false.
Proof.
  intros Hro Hu Hsrc from Hd.
  pose proof closed_S as Hc.
  destruct (s_has_source S from) eqn:Hhf; [ left | right; reflexivity ].
  destruct (alias_into_l is_local F T y _ u Hro from Hd)
    as [(b & Hb & Hub) | [(bs & Hs & Hbs) | (Hany & _)]].
  - apply (LP.least_holds is_local F T Hleast) in Hub. cbn [holds] in Hub.
    destruct (TP.usages_carry_sources is_local F T (or_introl Hleast) b u Hub)
      as [Hus Hua].
    destruct (s_any_source S b) eqn:Hab;
      [ rewrite (Hua eq_refl) in Hu; discriminate Hu | ].
    pose proof (sources_constructor _ _ _ Hc from _ b Hb Hab) as Hbb.
    destruct (Hus b Hbb) as [Hub' | Hua']; [ | rewrite Hua' in Hu;
                                               discriminate Hu ].
    exact (Hsrc b Hub' (has_source_constructor _ _ _ Hc from _ b Hhf Hb)
             from Hb).
  - apply (LP.least_holds is_local F T Hleast) in Hs. cbn [holds] in Hs.
    exact (Hsrc bs Hs (has_source_constructor _ _ _ Hc from _ bs Hhf Hbs)
             from Hbs).
  - apply (LP.least_holds is_local F T Hleast) in Hany. cbn [holds] in Hany.
    rewrite Hany in Hu. discriminate Hu.
Qed.

(* Such a read, an alias of [n], is dominated by [a0]. *)
Lemma read_dominated (y u : node) :
  LP.read_only F y (Field_value_slot w) u false ->
  s_any_source S u = false ->
  (∀ s, s_sources S u s = true -> s_has_source S s = true -> holds_n s) ->
  D (Atom_alias y n) ->
  mem_pair a0 y dominated = true.
Proof.
  intros Hro Hu Hsrc Hyn.
  pose proof closed_S as Hc.
  pose proof n_no_any_source as Hnany.
  destruct n_dominated as (Hna0 & Hha0 & Hnm).
  pose proof (read_aliases y u Hro Hu Hsrc) as Hk.
  assert (Hyany : s_any_source S y = false).
  { destruct (s_any_source S y) eqn:E; [ exfalso | reflexivity ].
    apply (LP.least_derivable is_local F T Hleast (Atom_any_source y)) in E.
    destruct (any_source_into_l is_local F T y _ u Hro E)
      as [(from & Hal & Haf) | Hau].
    - apply (LP.least_holds is_local F T Hleast) in Haf. cbn [holds] in Haf.
      destruct (Hk from Hal) as [-> | Hns];
        [ rewrite Haf in Hnany; discriminate Hnany | ].
      rewrite (has_source_any _ _ _ Hc from Haf) in Hns. discriminate Hns.
    - apply (LP.least_holds is_local F T Hleast) in Hau. cbn [holds] in Hau.
      rewrite Hau in Hu. discriminate Hu. }
  apply (proj2 (UP.check_dominated_sound T dominated Hdom a0 y)).
  split; [ | split; [ exact Hha0 | ] ].
  - apply (LP.least_holds is_local F T Hleast) in Hyn. cbn [holds] in Hyn.
    exact (sources_alias _ _ _ Hc n y a0 Hna0 Hnany Hyn Hyany).
  - assert (Hone : ∀ s, s_sources S y s = true -> s_has_source S s = true ->
                     s = a0).
    { intros s Hs Hhs.
      apply (LP.least_derivable is_local F T Hleast (Atom_sources y s)) in Hs.
      destruct (sources_into_l is_local F T y _ u Hro s Hs)
        as (from & Hfs & Hal).
      pose proof (sources_has_source from s Hfs Hhs) as Hhf.
      destruct (Hk from Hal) as [-> | Hns];
        [ | rewrite Hhf in Hns; discriminate Hns ].
      apply (LP.least_holds is_local F T Hleast) in Hfs. cbn [holds] in Hfs.
      destruct (node_eq_dec s a0) as [ | Hne]; [ assumption | exfalso ].
      apply Hnm. right. exists s, a0. auto. }
    intros [Hya | (s1 & s2 & Hs1 & Hh1 & Hs2 & Hh2 & Hne)];
      [ rewrite Hyany in Hya; discriminate Hya | ].
    apply Hne. rewrite (Hone s1 Hs1 Hh1), (Hone s2 Hs2 Hh2). reflexivity.
Qed.

(* The three ways a node is of [field_used_as] of an allocation. *)
Lemma field_used_as_cases (c y : node) (f : field) :
  y ∈ Ty.field_used_as ctx c f ->
  (∃ u, s_usages S c u = true ∧ g_accessor G y f u
        ∧ s_has_usage S y = true)
  ∨ (s_any_usage S c = true ∧ s_reading_field S f y = true)
  ∨ (is_local f = true ∧ s_any_usage S c = true
     ∧ ∃ u, s_sources S u c = true ∧ g_accessor G y f u).
Proof.
  intros Hy. unfold Ty.field_used_as in Hy.
  destruct (Ty.is_nil (Ty.constructors_of ctx c f)); [ destruct Hy | ].
  cbn [Ty.plain_context Ty.rc_is_local] in Hy.
  apply (proj1 (TP.In_dedup_nodes y _)) in Hy.
  apply in_app_or in Hy as [Hy | Hy];
    [ | apply in_app_or in Hy as [Hy | Hy] ].
  - left. apply in_flat_map in Hy as (u & Hu & Hy).
    apply TP.In_filter_has_use in Hy as [Hy Hhu].
    apply TP.usages_of_iff in Hu. apply TP.accessors_of_iff in Hy.
    exists u. auto.
  - right; left. destruct (Ty.any_usage ctx c) eqn:Hau; [ | destruct Hy ].
    apply in_map_iff in Hy as ([f' y'] & Hy' & Hin). cbn [snd] in Hy'.
    subst y'. apply filter_In in Hin as [Hin Hfy].
    apply andb_true_iff in Hfy as [Hf _]. cbn [fst] in Hf.
    apply field_eqb_eq in Hf. subst f'.
    split; [ exact Hau | ]. cbn [solution_of_tables s_reading_field].
    apply mem_field_node_In. exact Hin.
  - right; right.
    destruct (is_local f && Ty.any_usage ctx c) eqn:Hla; [ | destruct Hy ].
    apply andb_true_iff in Hla as [Hloc Hau].
    apply in_flat_map in Hy as (u & Hu & Hy).
    apply TP.In_filter_has_use in Hy as [Hy _].
    apply (TP.In_filter_map_fst u c) in Hu. apply TP.accessors_of_iff in Hy.
    split; [ exact Hloc | split; [ exact Hau | ] ].
    exists u. split; [ | exact Hy ].
    cbn [solution_of_tables s_sources]. apply mem_pair_In. exact Hu.
Qed.

(* A read of the field from the closures with a use is dominated by
   [a0]. *)
Lemma case_read (c y : node) :
  c ∈ map snd (set_function_slots F x0) ->
  y ∈ Ty.field_used_as ctx c (Field_value_slot w) ->
  Ty.field_used_top ctx c (Field_value_slot w) = false ->
  mem_pair a0 y dominated = true.
Proof.
  intros Hc Hy Htop.
  pose proof closed_S as Hc0.
  destruct (Hcl c Hc) as [Hcn (g & Hg)].
  assert (Hcons : g_constructor G c (Field_value_slot w) n)
    by (apply Hcn; reflexivity).
  destruct (TP.field_used_as_alias is_local F T (or_introl Hleast) [] c n y
              _ Hy Hcons Htop)
    as (Hhu & _ & Hal).
  apply (LP.least_derivable is_local F T Hleast (Atom_alias y n)) in Hal.
  pose proof (closure_not_crr0 c Hc) as Hcrr0.
  assert (Hhc : s_has_source S c = true)
    by exact (has_source_function_slot _ _ _ Hc0 x0 _ c Hg eq_refl).
  assert (Hholds : holds_n c) by (intros b Hb; apply Hcn; exact Hb).
  destruct (field_used_as_cases c y _ Hy)
    as [(u & Hu & Hacc & Hhy) | [(Hau & Hrf) | (Hloc & Hau & u & Hsu & Hacc)]].
  - pose proof (Hread y u Hacc) as Hro.
    apply (read_dominated y u Hro); [ | | exact Hal ].
    + destruct (s_any_source S u) eqn:E; [ exfalso | reflexivity ].
      rewrite (US.crr0_any_source_usage _ _ _ _ _ _ _ _ _ _ _ Hclosed c u
                 (Field_value_slot w) y Hu E eq_refl Hacc Hhy)
        in Hcrr0.
      discriminate Hcrr0.
    + intros s Hs Hhs.
      destruct (node_eq_dec s c) as [-> | Hne]; [ exact Hholds | exfalso ].
      rewrite (US.crr0_other_source _ _ _ _ _ _ _ _ _ _ _ Hclosed c u s
                 (Field_value_slot w) y Hu Hs Hhs Hne eq_refl Hacc Hhy)
        in Hcrr0.
      discriminate Hcrr0.
  - exfalso.
    assert (Hloc : is_local (Field_value_slot w) = true).
    { apply (LP.least_derivable is_local F T Hleast
               (Atom_reading_field _ y)) in Hrf.
      inversion Hrf; subst. assumption. }
    rewrite (US.crr0_local_reading _ _ _ _ _ _ _ _ _ _ _ Hclosed c _ n y Hau
               Hloc Hrf Hcons)
      in Hcrr0.
    discriminate Hcrr0.
  - pose proof (Hread y u Hacc) as Hro.
    apply (read_dominated y u Hro); [ | | exact Hal ].
    + exact (TP.sources_not_any is_local F T (or_introl Hleast) u c Hsu).
    + intros s Hs Hhs.
      destruct (node_eq_dec s c) as [-> | Hne]; [ exact Hholds | exfalso ].
      rewrite (US.crr0_local_sources _ _ _ _ _ _ _ _ _ _ _ Hclosed u _ y c s
                 Hacc Hhu Hloc Hsu Hhc Hs Hhs (not_eq_sym Hne))
        in Hcrr0.
      discriminate Hcrr0.
Qed.

(* A read of the field with any usage would give [n] any usage, and so
   [a0]. *)
Lemma case_unknown (c : node) :
  c ∈ map snd (set_function_slots F x0) ->
  Ty.field_used_top ctx c (Field_value_slot w) = true -> False.
Proof.
  intros Hc Htop. destruct (Hcl c Hc) as [Hcn _].
  pose proof (TP.field_used_top_any_usage is_local F T (or_introl Hleast) []
                c n _ Htop (proj2 (Hcn n) eq_refl))
    as Hn.
  destruct n_dominated as (Hna0 & _ & _).
  apply (LP.least_derivable is_local F T Hleast (Atom_sources n a0)) in Hna0.
  pose proof (LP.sources_any_usage is_local F T Hleast n a0 Hna0 Hn) as H.
  pose proof a0_no_any_usage as H'. unfold_rels.
  rewrite H' in H. discriminate H.
Qed.

(* [a0] is the one source of the field. *)
Lemma field_source_a0 :
  (∃ from, g_constructor G x0 (Field_value_slot w) from) ->
  get_single_field_source F T x0 (Field_value_slot w) = One_source a0.
Proof.
  intros (from0 & Hfrom0).
  pose proof n_no_any_source as Hnany.
  destruct n_dominated as (Hna0 & Hha0 & Hnm).
  unfold get_single_field_source.
  destruct (existsb _ (f_constructor F)) eqn:Hex.
  { exfalso. apply existsb_exists in Hex as ([[b f'] s] & Hin & Hb).
    apply andb_true_iff in Hb as [Hb Has].
    apply andb_true_iff in Hb as [Hb Hf].
    apply node_eqb_eq in Hb. apply field_eqb_eq in Hf. subst b f'.
    rewrite (proj1 (Hx0 s) Hin) in Has. unfold_rels.
    rewrite Has in Hnany. discriminate Hnany. }
  rewrite (dedup_single _ a0); [ reflexivity | | ].
  - apply in_flat_map. exists (x0, Field_value_slot w, from0).
    split; [ exact Hfrom0 | ].
    rewrite node_eqb_refl, field_eqb_refl. cbn [andb].
    rewrite (proj1 (Hx0 from0) Hfrom0).
    apply in_flat_map. exists (n, a0).
    split; [ apply mem_pair_In; exact Hna0 | ].
    rewrite node_eqb_refl. unfold_rels. rewrite Hha0. left. reflexivity.
  - intros z Hz. apply in_flat_map in Hz as ([[b f'] fs] & Hin & Hz).
    destruct (node_eqb b x0 && field_eqb (Field_value_slot w) f') eqn:Hbf;
      [ | destruct Hz ].
    apply andb_true_iff in Hbf as [Hb Hf].
    apply node_eqb_eq in Hb. apply field_eqb_eq in Hf. subst b f'.
    rewrite (proj1 (Hx0 fs) Hin) in Hz.
    apply in_flat_map in Hz as ([n' s] & Hs & Hz).
    destruct (node_eqb n n' && mem_node s (t_has_source T)) eqn:Hns;
      [ | destruct Hz ].
    destruct Hz as [<- | []].
    apply andb_true_iff in Hns as [Hn' Hhs]. apply node_eqb_eq in Hn'.
    subst n'.
    destruct (node_eq_dec s a0) as [ | Hne]; [ assumption | exfalso ].
    apply Hnm. right. exists s, a0. unfold_rels.
    split; [ apply mem_pair_In; exact Hs | ].
    split; [ exact Hhs | ]. split; [ exact Hna0 | ]. auto.
Qed.

Lemma depth_pos : UF.unboxing_depth T <> 0.
Proof.
  destruct n_dominated as (Hna0 & _ & _). unfold_rels.
  apply mem_pair_In in Hna0. unfold UF.unboxing_depth.
  destruct (t_sources T) as [ | p l]; [ destruct Hna0 | ].
  apply Nat.neq_mul_0. split; [ discriminate | ].
  apply Nat.pow_nonzero. discriminate.
Qed.

(* The step of the field cannot leave it a variable. *)
Lemma not_unboxed_absurd (use : field_use) (cm : finite_map variable)
    (next next' : positive) (v : variable) :
  (∃ from, g_constructor G x0 (Field_value_slot w) from) ->
  (Field_value_slot w, use)
    ∈ get_fields_of_constructors F T is_local
        (map snd (set_function_slots F x0)) ->
  URP.field_step F T cm dominated to_unbox (UF.unboxing_depth T) x0
    (Field_value_slot w) use next
  = (Some (Not_unboxed v), next') ->
  False.
Proof.
  intros Hex Huse Hstep.
  pose proof (field_source_a0 Hex) as Hsrc.
  pose proof depth_pos as Hdp.
  destruct (TP.gfoc_spec is_local F T [] _ _ _ Huse) as [_ Hu].
  unfold URP.field_step in Hstep. cbn [is_data_field negb] in Hstep.
  rewrite Hsrc in Hstep. cbv beta iota zeta in Hstep.
  destruct use as [ | flow_to].
  - destruct Hu as (c & Hc & Htop). exact (case_unknown c Hc Htop).
  - destruct Hu as (_ & Hnot & Hiff).
    destruct (forallb (has_to_be_unboxed dominated to_unbox) flow_to)
      eqn:Hall.
    + destruct (UF.unboxing_depth T) as [ | k]; [ exact (Hdp eq_refl) | ].
      destruct (mk_unboxed_fields _ _ _ _ _ _ _ _ _) in Hstep.
      discriminate Hstep.
    + apply not_true_iff_false in Hall. apply Hall.
      apply forallb_forall. intros y Hy.
      destruct (proj1 (Hiff y) Hy) as (c & Hc & Hyc).
      pose proof (case_read c y Hc Hyc (Hnot c Hc)) as Hd.
      unfold has_to_be_unboxed. apply existsb_exists. exists (a0, y).
      split; [ apply mem_pair_In; exact Hd | ].
      rewrite node_eqb_refl, Ha0u. reflexivity.
Qed.

End Plain.

(* ======================================================================== *)
(* The field of a [Let] of a closure                                        *)
(* ======================================================================== *)

Theorem changed_field_plain : changed_field_plain_statement.
Proof.
  intros is_local u e T cannot_change max_unbox_size lambda_lifting dominated
    to_unbox tcr x slot decls vs r w v a Hflow Hwfu Honce Hleast Htables Hx
    Hlk Hfu Hw.
  set (F := traverse_unit Auto u e) in *.
  pose proof Htables as Ht. unfold analysis_tables in Ht.
  apply andb_true_iff in Ht as [Ht _].
  apply andb_true_iff in Ht as [Hdom Hfix].
  destruct (UP.check_to_change_representation_fixed_sound is_local F T
              cannot_change max_unbox_size lambda_lifting to_unbox tcr Hfix)
    as (crr0 & crr1 & crr & cu0 & cu & Hclosed & _ & Hcu & _ & Htcr & _).
  pose proof (TP.least_closed_tables is_local F T (or_introl Hleast)) as Hc.
  (* The representation is that of the set of an allocation [x0] of
     [tcr], of which [x] is a closure. *)
  pose proof (CTP.closure_let_self u e x slot decls vs Hflow Hwfu Hx) as Hxx.
  pose proof Hlk as Hlk'. unfold unit_changed_representations in Hlk'.
  apply (keyed_lookup_in node_eqb TP.node_eqb_decides) in Hlk'.
  destruct (FTP.reprs_in _ _ _ _ _ _ _ _ _ _ Hlk')
    as (x0 & f0 & g & n' & Hx0 & Hg & Hd & Hfields).
  rewrite (TTP.dominated_self is_local F T dominated Hc Hdom f0
             (Node_variable x) (Node_variable x) slot Hxx Hd)
    in Hg.
  apply TP.set_function_slots_in in Hg.
  rewrite Hfields in Hfu.
  destruct (mk_step_in _ _ _ _ _ _ _ _ _ _ _ Hfu)
    as (use & n1 & n2 & Huse & Hstep).
  destruct (FTP.mk_field_source _ _ _ _ _ _ _ _ _ _ _ Hfu) as [_ Hsrc].
  destruct (FTP.source_constructor _ _ _ _ Hsrc) as (from & Hfrom).
  (* The closures of the set hold at [w] what [x] holds: [a] alone. *)
  destruct (STP.let_node_exact u e x slot decls vs Hflow Hwfu Honce Hx)
    as (_ & _ & E3 & _).
  pose proof (FTP.sets_of_traversal u e Hflow Hwfu Honce) as Hsets.
  destruct (TTP.function_slots_mutual_traversal u e Hflow Hwfu Honce x0
              (Node_variable x) g Hg)
    as (g0 & Hxx0).
  assert (Hxw : ∀ b, g_constructor (graph_of_facts F) (Node_variable x)
                       (Field_value_slot w) b
                     <-> b = node_of_simple a).
  { intros b. split.
    - intros Hb.
      destruct (STP.closure_value_slots Auto u e x slot decls vs w b Hflow
                  Honce Hx Hb)
        as (a' & Ha' & ->).
      rewrite Hw in Ha'. injection Ha' as <-. reflexivity.
    - intros ->.
      pose proof (proj1 (proj2 (E3 g0 x0 Hxx0) w from) Hfrom) as Hxf.
      destruct (STP.closure_value_slots Auto u e x slot decls vs w from Hflow
                  Honce Hx Hxf)
        as (a' & Ha' & Ef).
      rewrite Hw in Ha'. injection Ha' as <-. rewrite <- Ef. exact Hxf. }
  assert (Hx0w : ∀ b, g_constructor (graph_of_facts F) x0
                        (Field_value_slot w) b
                      <-> b = node_of_simple a).
  { intros b. split.
    - intros H. apply Hxw. exact (proj1 (proj2 (E3 g0 x0 Hxx0) w b) H).
    - intros H. exact (proj2 (proj2 (E3 g0 x0 Hxx0) w b) (proj2 (Hxw b) H)). }
  assert (Hcl :
            ∀ c, c ∈ map snd (set_function_slots F x0) ->
              (∀ b, g_constructor (graph_of_facts F) c (Field_value_slot w) b
                    <-> b = node_of_simple a)
              ∧ ∃ g' : function_slot,
                  g_constructor (graph_of_facts F) c (Field_function_slot g')
                    x0).
  { intros c Hcin.
    destruct (FTP.closure_of_x u e x Hflow Hwfu Honce x0 c g Hg Hcin)
      as (g' & Hxc).
    split.
    - intros b. split.
      + intros H. apply Hxw. exact (proj1 (proj2 (E3 g' c Hxc) w b) H).
      + intros H. exact (proj2 (proj2 (E3 g' c Hxc) w b) (proj2 (Hxw b) H)).
    - exists g0.
      exact (proj1 ((proj2 (proj2 Hsets)) (Node_variable x) c x0 g' g0 Hxc)
               Hxx0). }
  (* The reads of the slot are only read into. *)
  assert (Hread : ∀ y b, g_accessor (graph_of_facts F) y (Field_value_slot w) b
                    -> LP.read_only F y (Field_value_slot w) b false).
  { intros y b Hacc.
    destruct (TSP.slot_reads_projected Auto u e y w b Honce Hacc)
      as (v' & d & c' & -> & Hd' & Hs & ->).
    apply LP.only_read_into_read_only.
    exact (TSP.slot_read_only Auto u e v' d w c' Honce Hd' Hs). }
  assert (Hcrr : crr x0 = false)
    by exact (proj1 (proj2 (Htcr x0 (proj2 (mem_node_In _ _) Hx0)))).
  (* An allocation to unbox that dominates [a] would unbox the field. *)
  assert (Hnot : ∀ a0, (a0, node_of_simple a) ∈ dominated ->
                   mem_node a0 to_unbox = true -> False).
  { intros a0 Ha0 Ha0u.
    exact (not_unboxed_absurd is_local F T Hleast crr0 crr1 crr cu0 cu
             cannot_change max_unbox_size lambda_lifting Hclosed to_unbox
             (fun x' H => proj2 (Hcu x' H)) dominated Hdom x0 Hcrr w
             (node_of_simple a) Hx0w Hcl Hread a0 Ha0 Ha0u use _ n1 n2 v
             (ex_intro _ from Hfrom) Huse Hstep). }
  destruct a as [b | sym | k]; cbn [plain_simple node_of_simple] in *;
    [ | | reflexivity ].
  - unfold plain_variable, Keyed.is_bound.
    destruct (Keyed.lookup node_eqb (Node_variable b)
                (fst (unit_unboxed_fields F T dominated to_unbox e)))
      as [fs | ] eqn:El; [ exfalso | reflexivity ].
    apply (keyed_lookup_in node_eqb TP.node_eqb_decides) in El.
    unfold unit_unboxed_fields in El.
    destruct (URP.unboxed_fields_of_names_dom _ _ _ _ _ _ _ _ El)
      as (a0 & _ & Ha0 & Ha0u & _).
    exact (Hnot a0 Ha0 Ha0u).
  - unfold Keyed.is_bound.
    destruct (Keyed.lookup node_eqb (Node_symbol sym)
                (fst (unit_unboxed_fields F T dominated to_unbox e)))
      as [fs | ] eqn:El; [ exfalso | reflexivity ].
    apply (keyed_lookup_in node_eqb TP.node_eqb_decides) in El.
    unfold unit_unboxed_fields in El.
    destruct (URP.unboxed_fields_of_names_dom _ _ _ _ _ _ _ _ El)
      as (a0 & _ & Ha0 & Ha0u & _).
    exact (Hnot a0 Ha0 Ha0u).
Qed.

(* ======================================================================== *)
(* The fragment                                                             *)
(* ======================================================================== *)

Theorem repr_unboxing_frag4_sound : repr_unboxing_frag4_sound_statement.
Proof.
  intros nr is_local u e T cannot_change max_unbox_size lambda_lifting
    code_deps dominated to_unbox tcr Hflow Hwfu Honce Hleast Htables H.
  unfold repr_unboxing_frag4_nr in H. unfold repr_unboxing_frag3_nr.
  apply andb_true_iff in H as [H Huses]. rewrite Huses, andb_true_r.
  refine (proj1 (STP.with_mono _ _ _ _ _ _ _ _ _ _ (let_bindings e) _ _ _) e
            [] [] (fun p Hp => Hp) H).
  intros x kind d body r Hx Hch Hfrag.
  destruct d as [s | p args | id vs | slot decls vs | w c | pf w c | mf mt c];
    cbn [changed_set_frag4_ok] in Hfrag; try discriminate Hfrag.
  unfold changed_name, changed_of_node in Hch.
  cbn [changed_set_frag3_ok].
  apply andb_true_iff in Hfrag as [Hfrag Hfields].
  rewrite Hfrag. cbn [andb].
  rewrite forallb_forall in Hfields |- *. intros [f uf] Hfu.
  specialize (Hfields _ Hfu).
  destruct f as [i k | w | fs | | | | cw | rc | ]; try exact Hfields.
  destruct uf as [v | fsu]; [ | exact Hfields ].
  cbn [repr_field_frag4_ok] in Hfields. cbn [repr_field_frag3_ok].
  destruct (lookup w vs) as [a | ] eqn:Hw; [ | reflexivity ].
  cbn [repr_field_frag_ok]. rewrite Hw. unfold plain_c.
  rewrite Hfields, andb_true_r.
  exact (changed_field_plain is_local u e T cannot_change max_unbox_size
           lambda_lifting dominated to_unbox tcr x slot decls vs r w v a
           Hflow Hwfu Honce Hleast Htables Hx Hch Hfu Hw).
Qed.

(* ======================================================================== *)
(* The refinement                                                           *)
(* ======================================================================== *)

Theorem rebuild_repr_frag4_refines_cc :
  rebuild_repr_frag4_refines_cc_statement.
Proof.
  intros is_local u e T cannot_change max_unbox_size lambda_lifting dominated
    to_unbox tcr pdc nr Hloc Hflow Hwfu Honce Hleast Hcc Htables Hfrag Hreads.
  exact (STP.rebuild_repr_frag3_refines_cc is_local u e T cannot_change
           max_unbox_size lambda_lifting dominated to_unbox tcr pdc nr Hloc
           Hflow Hwfu Honce Hleast Hcc Htables
           (repr_unboxing_frag4_sound nr is_local u e T cannot_change
              max_unbox_size lambda_lifting _ dominated to_unbox tcr Hflow
              Hwfu Honce Hleast Htables Hfrag)
           Hreads).
Qed.

Theorem rebuild_repr_frag4_observably :
  rebuild_repr_frag4_observably_statement.
Proof.
  intros is_local u e T cannot_change max_unbox_size lambda_lifting dominated
    to_unbox tcr pdc nr Hloc Hproj Hflow Hany Hres Hwfu Honce Hexits Hleast
    Hcc Hexact Hccc Hwidgets Htables Hfrag Hreads.
  exact (STP.rebuild_repr_frag3_observably is_local u e T cannot_change
           max_unbox_size lambda_lifting dominated to_unbox tcr pdc nr Hloc
           Hproj Hflow Hany Hres Hwfu Honce Hexits Hleast Hcc Hexact Hccc
           Hwidgets Htables
           (repr_unboxing_frag4_sound nr is_local u e T cannot_change
              max_unbox_size lambda_lifting _ dominated to_unbox tcr Hflow
              Hwfu Honce Hleast Htables Hfrag)
           Hreads).
Qed.

Theorem rebuild_repr_frag4_local : rebuild_repr_frag4_local_statement.
Proof.
  intros is_local u e T cannot_change max_unbox_size lambda_lifting dominated
    to_unbox tcr pdc nr Hloc Hflow Hany Hres Hwfu Honce Hexits Hleast Hcc
    Hlocal Hexact Hccc Hwidgets Htables Hfrag Hreads.
  exact (STP.rebuild_repr_frag3_local is_local u e T cannot_change
           max_unbox_size lambda_lifting dominated to_unbox tcr pdc nr Hloc
           Hflow Hany Hres Hwfu Honce Hexits Hleast Hcc Hlocal Hexact Hccc
           Hwidgets Htables
           (repr_unboxing_frag4_sound nr is_local u e T cannot_change
              max_unbox_size lambda_lifting _ dominated to_unbox tcr Hflow
              Hwfu Honce Hleast Htables Hfrag)
           Hreads).
Qed.

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

(* The proofs of FlambdaReaperIndirectDeadFramesSpec.v. Nothing here needs
   review.

   [covers P x]: [x] has no any source, and each source of [x] with a
   source is of [P]. A node only read off a covered node is covered by
   what [get_field_sources] computes ([read_covers]): an alias into it
   comes from a constructor that is a source of the base, directly or,
   for a usage of the base, by [usages_flow]. A node only aliased to a
   covered node is covered alike ([alias_covers]). Along the chain of
   witnesses of the call, the sets that [never_returns_of_call] reaches
   cover each witness ([chain_dead]); at the last, a result of rank [i]
   with a source would make [result_has_source_of] true. So
   [apply_never_returns] gives [call_dead_rank] ([never_returns_dead]),
   and [marked_indirect_run_at] the statement. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaValueSpec FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperSolutionSpec FlambdaReaperSolutionProofs
  FlambdaReaperFlowSpec FlambdaReaperFlowProofs FlambdaReaperRebuild
  FlambdaReaperCallingConventions FlambdaReaperUnboxing
  FlambdaReaperLeastSpec FlambdaReaperDeadFramesSpec
  FlambdaReaperDeadFramesProofs FlambdaReaperIndirectDeadFramesSpec.
From OxCamlFlambda2 Require FlambdaReaperLeast FlambdaReaperLeastProofs
  FlambdaReaperWrapperProofs.
Import ListNotations.

Module LP := FlambdaReaperLeastProofs.

Local Abbreviation unknown := (Field_call_witness Unknown_arity_code_pointer).
Local Abbreviation first_result := (Field_return_of_call (Normal 0)).

Ltac unfold_rels :=
  cbn [graph_of_facts solution_of_tables holds g_alias g_use g_accessor
       g_constructor g_argument g_parameter g_propagate
       g_alias_if_any_source g_any_usage g_any_source g_any_source_if_used
       g_zero_alloc_source s_alias s_any_usage s_any_source s_has_usage
       s_has_source s_usages s_sources s_reading_field s_escaping_field
       s_field_used] in *.

(* ======================================================================== *)
(* The checks, read back                                                    *)
(* ======================================================================== *)

(* What [only_alias_into F x from] says, fact by fact. *)
Record alias_only (F : facts) (x from : node) : Prop := {
  ao_alias : ∀ y, (x, y) ∈ f_alias F -> y = from;
  ao_propagate : ∀ c y, (c, x, y) ∈ f_propagate F -> y = from;
  ao_if_any_source : ∀ c y, (c, x, y) ∉ f_alias_if_any_source F;
  ao_accessor : ∀ f b, (x, f, b) ∉ f_accessor F;
  ao_parameter_to : ∀ b i, (b, i, x) ∉ f_parameter F;
  ao_parameter_base : ∀ i t, (x, i, t) ∉ f_parameter F;
  ao_constructor : ∀ r y, (x, r, y) ∉ f_constructor F;
  ao_any_source : x ∉ f_any_source F;
  ao_zero_alloc : x ∉ f_zero_alloc_source F;
  ao_if_used : ∀ c, (c, x) ∉ f_any_source_if_used F;
  ao_use : ∀ y, (x, y) ∉ f_use F
}.

Lemma only_alias_into_alias_only (F : facts) (x from : node) :
  only_alias_into F x from = true -> alias_only F x from.
Proof.
  unfold only_alias_into. intros H.
  apply andb_true_iff in H as [H Huse].
  apply andb_true_iff in H as [H Hused].
  apply andb_true_iff in H as [H Hzero].
  apply andb_true_iff in H as [H Hany].
  apply andb_true_iff in H as [H Hcons].
  apply andb_true_iff in H as [H Hparam].
  apply andb_true_iff in H as [H Hacc].
  apply andb_true_iff in H as [H Hias].
  apply andb_true_iff in H as [Halias Hprop].
  rewrite forallb_forall in Halias, Hprop, Hias, Hacc, Hparam, Hcons,
    Hused, Huse.
  apply negb_true_iff in Hany, Hzero.
  constructor.
  - intros y Hin. specialize (Halias _ Hin). cbn in Halias.
    rewrite node_eqb_refl in Halias. cbn in Halias.
    apply node_eqb_eq in Halias. exact Halias.
  - intros c y Hin. specialize (Hprop _ Hin). cbn in Hprop.
    rewrite node_eqb_refl in Hprop. cbn in Hprop.
    apply node_eqb_eq in Hprop. exact Hprop.
  - intros c y Hin. specialize (Hias _ Hin). cbn in Hias.
    rewrite node_eqb_refl in Hias. discriminate.
  - intros f b Hin. specialize (Hacc _ Hin). cbn in Hacc.
    rewrite node_eqb_refl in Hacc. discriminate.
  - intros b i Hin. specialize (Hparam _ Hin). cbn in Hparam.
    rewrite node_eqb_refl in Hparam. discriminate.
  - intros i t Hin. specialize (Hparam _ Hin). cbn in Hparam.
    rewrite node_eqb_refl, orb_true_r in Hparam. discriminate.
  - intros r y Hin. specialize (Hcons _ Hin). cbn in Hcons.
    rewrite node_eqb_refl in Hcons. discriminate.
  - intros Hin. apply mem_node_In in Hin. congruence.
  - intros Hin. apply mem_node_In in Hin. congruence.
  - intros c Hin. specialize (Hused _ Hin). cbn in Hused.
    rewrite node_eqb_refl in Hused. discriminate.
  - intros y Hin. specialize (Huse _ Hin). cbn in Huse.
    rewrite node_eqb_refl in Huse. discriminate.
Qed.

(* What [check_unknown_chains] says of a read [x] of the call witness of
   unknown arity off [n]. *)
Lemma check_unknown_chains_at (F : facts) (x n : node) :
  check_unknown_chains F = true -> (x, unknown, n) ∈ f_accessor F ->
  LP.read_only F x unknown n false
  ∧ ((∃ c, (c, Field_code_id_of_call_witness, x) ∈ f_accessor F) ->
     ∃ w, (n, first_result, w) ∈ f_accessor F
          ∧ LP.read_only F n first_result w false)
  ∧ (∀ w, (w, x) ∈ f_alias F -> alias_only F w x)
  ∧ (∀ c w, (c, w, x) ∈ f_propagate F -> alias_only F w x).
Proof.
  intros Hcheck Hin. unfold check_unknown_chains in Hcheck.
  rewrite forallb_forall in Hcheck. specialize (Hcheck _ Hin).
  cbn beta iota in Hcheck. cbn [negb orb] in Hcheck.
  apply andb_true_iff in Hcheck as [H Hprop].
  apply andb_true_iff in H as [H Halias].
  apply andb_true_iff in H as [Hro Hwit].
  split; [ exact (LP.only_read_into_read_only _ _ _ _ _ Hro) | ].
  split; [ | split ].
  - intros (c & Hc).
    apply orb_true_iff in Hwit as [Hno | Hyes].
    + exfalso. apply negb_true_iff in Hno.
      apply Bool.not_true_iff_false in Hno. apply Hno.
      apply existsb_exists. exists (c, Field_code_id_of_call_witness, x).
      split; [ exact Hc | ]. cbn.
      rewrite node_eqb_refl. reflexivity.
    + apply existsb_exists in Hyes as (((y, f'), w) & Hy & Hw).
      apply andb_true_iff in Hw as [Hw Hro'].
      apply andb_true_iff in Hw as [Hyn Hf'].
      apply node_eqb_eq in Hyn. apply field_eqb_eq in Hf'. subst y f'.
      exists w. split; [ exact Hy | ].
      exact (LP.only_read_into_read_only _ _ _ _ _ Hro').
  - intros w Hw. rewrite forallb_forall in Halias.
    specialize (Halias _ Hw). cbn in Halias.
    rewrite node_eqb_refl in Halias. cbn in Halias.
    exact (only_alias_into_alias_only _ _ _ Halias).
  - intros c w Hw. rewrite forallb_forall in Hprop.
    specialize (Hprop _ Hw). cbn in Hprop.
    rewrite node_eqb_refl in Hprop. cbn in Hprop.
    exact (only_alias_into_alias_only _ _ _ Hprop).
Qed.

Lemma not_local_field (is_local : field -> bool) (f : field) :
  is_local_slots_only is_local ->
  (∀ w : value_slot, f <> Field_value_slot w) ->
  (∀ fs : function_slot, f <> Field_function_slot fs) ->
  is_local f = false.
Proof.
  intros Hlocal Hw Hf. destruct (is_local f) eqn:E; [ | reflexivity ].
  destruct (Hlocal f E) as [(w & ->) | (fs & ->)];
    [ destruct (Hw w) | destruct (Hf fs) ]; reflexivity.
Qed.

(* ======================================================================== *)
(* Covered nodes                                                            *)
(* ======================================================================== *)

Section Chains.

Variable is_local : field -> bool.
Variable F : facts.
Variable T : tables.
Hypothesis Hleast : least_solution is_local F T.

Local Abbreviation G := (graph_of_facts F).
Local Abbreviation S := (solution_of_tables T).
Local Abbreviation D := (derivable is_local G S).

(* [x] has no any source, and its sources with a source are of [P]. *)
Definition covers (P : node -> bool) (x : node) : Prop :=
  s_any_source S x = false
  ∧ ∀ s : node,
      s_sources S x s = true -> s_has_source S s = true -> P s = true.

(* A source with a source gives one to the node. *)
Lemma sources_has_source (x s : node) :
  D (Atom_sources x s) -> s_has_source S s = true -> s_has_source S x = true.
Proof.
  pose proof (LP.least_closed is_local F T Hleast) as Hc.
  intros Hd. remember (Atom_sources x s) as a eqn:E. revert x s E.
  induction Hd; intros x0 s0 E Hs; try discriminate E; injection E as <- <-.
  - exact Hs.
  - exact Hs.
  - apply (LP.least_holds is_local F T Hleast) in Hd2.
    exact (has_source_alias _ _ _ Hc from to_
             (IHHd1 from source eq_refl Hs) Hd2).
Qed.

(* A node read off [base] with [f], into which nothing else puts
   anything, has the sources that [get_field_sources] computes. *)
Lemma read_covers (x base : node) (f : field) (P P' : node -> bool) :
  LP.read_only F x f base false -> is_local f = false ->
  covers P base -> get_field_sources F S P f = Sources P' -> covers P' x.
Proof.
  pose proof (LP.least_closed is_local F T Hleast) as Hc.
  intros Hro Hnl [Hbany HbP] Hget.
  unfold get_field_sources in Hget.
  match type of Hget with
  | (if existsb ?p (f_constructor F) then _ else _) = _ =>
    destruct (existsb p (f_constructor F)) eqn:Eany; [ discriminate Hget | ]
  end.
  injection Hget as <-.
  (* A constructor at [f] reached from [base], whose field has a
     source, is of [P]. *)
  assert (Hsrc : ∀ b from : node,
             (b, f, from) ∈ f_constructor F -> s_has_source S from = true ->
             (D (Atom_usages b base) ∨ D (Atom_sources base b)) ->
             P b = true).
  { intros b from Hb Hhf Hub.
    assert (Hhb : s_has_source S b = true)
      by exact (has_source_constructor _ _ _ Hc from f b Hhf Hb).
    apply (HbP b); [ | exact Hhb ].
    destruct Hub as [Hu | Hs].
    - destruct (LP.usages_flow is_local F T Hleast b base Hu)
        as [Hany Hsrcs].
      destruct (s_any_source S b) eqn:Eb.
      + unfold_rels. rewrite (Hany Eb) in Hbany. discriminate.
      + pose proof (sources_constructor _ _ _ Hc from f b Hb Eb) as Hbb.
        unfold_rels.
        destruct (Hsrcs b Hbb) as [H | H]; [ congruence | exact H ].
    - exact (LP.least_holds is_local F T Hleast _ Hs). }
  (* The two ways that an alias gets into [x]. *)
  assert (Hal : ∀ from : node,
             D (Atom_alias x from) -> s_has_source S from = true ->
             ∃ b : node, (b, f, from) ∈ f_constructor F ∧ P b = true).
  { intros from Hd Hhf.
    destruct (LP.alias_into is_local F T x f base false Hro Hnl from Hd)
      as [(b & Hb & Hu & _) | [(bs & Hs & _ & Hb) | [Hf _]]];
      [ | | discriminate Hf ].
    - exists b. split; [ exact Hb | ].
      exact (Hsrc b from Hb Hhf (or_introl Hu)).
    - exists bs. split; [ exact Hb | ].
      exact (Hsrc bs from Hb Hhf (or_intror Hs)). }
  split.
  - destruct (s_any_source S x) eqn:Ex; [ exfalso | reflexivity ].
    apply (LP.least_derivable is_local F T Hleast (Atom_any_source x)) in Ex.
    destruct (LP.any_source_into is_local F T x f base false Hro Ex)
      as [(from & Hd & Hf) | Hb].
    + apply (LP.least_holds is_local F T Hleast) in Hf. cbn [holds] in Hf.
      destruct (Hal from Hd (has_source_any _ _ _ Hc from Hf))
        as (b & Hb & HPb).
      apply Bool.not_true_iff_false in Eany. apply Eany.
      apply existsb_exists. exists (b, f, from). split; [ exact Hb | ].
      rewrite HPb, field_eqb_refl, Hf. reflexivity.
    + apply (LP.least_holds is_local F T Hleast) in Hb. cbn [holds] in Hb.
      congruence.
  - intros s Hs Hhs.
    apply (LP.least_derivable is_local F T Hleast (Atom_sources x s)) in Hs.
    destruct (LP.sources_into is_local F T x f base false Hro s Hs)
      as (from & Hfs & _ & Hd & _).
    pose proof (sources_has_source from s Hfs Hhs) as Hhf.
    destruct (Hal from Hd Hhf) as (b & Hb & HPb).
    apply existsb_exists. exists (b, f, from). split; [ exact Hb | ].
    apply (LP.least_holds is_local F T Hleast) in Hfs. unfold_rels.
    rewrite HPb, field_eqb_refl, Hfs, Hhs. reflexivity.
Qed.

(* Under [alias_only F w x], every alias of [w] is one of [x]. *)
Lemma alias_only_alias (w x : node) :
  alias_only F w x -> ∀ from : node, D (Atom_alias w from) -> from = x.
Proof.
  intros Hao from Hd. inversion Hd; subst; unfold_rels;
    try solve
      [ eapply (ao_alias _ _ _ Hao); eassumption
      | eapply (ao_propagate _ _ _ Hao); eassumption
      | exfalso; eapply (ao_if_any_source _ _ _ Hao); eassumption
      | exfalso; eapply (ao_accessor _ _ _ Hao); eassumption
      | exfalso; eapply (ao_parameter_to _ _ _ Hao); eassumption ].
  match goal with
  | H : derivable _ _ _ (Atom_reading_field _ _) |- _ =>
    inversion H; subst; unfold_rels
  end.
  exfalso. eapply (ao_accessor _ _ _ Hao). eassumption.
Qed.

(* A node only aliased to [x] is covered as [x] is. *)
Lemma alias_covers (w x : node) (P : node -> bool) :
  alias_only F w x -> covers P x -> covers P w.
Proof.
  intros Hao [Hxany HxP]. split.
  - destruct (s_any_source S w) eqn:E; [ exfalso | reflexivity ].
    apply (LP.least_derivable is_local F T Hleast (Atom_any_source w)) in E.
    inversion E; subst; unfold_rels;
      try solve
        [ eapply (ao_any_source _ _ _ Hao); eassumption
        | eapply (ao_zero_alloc _ _ _ Hao); eassumption
        | eapply (ao_if_used _ _ _ Hao); eassumption
        | eapply (ao_parameter_to _ _ _ Hao); eassumption
        | eapply (ao_accessor _ _ _ Hao); eassumption
        | eapply (ao_use _ _ _ Hao); eassumption ].
    match goal with
    | H : derivable _ _ _ (Atom_alias _ ?from),
      H' : derivable _ _ _ (Atom_any_source ?from) |- _ =>
      rewrite (alias_only_alias w x Hao from H) in H';
      apply (LP.least_holds is_local F T Hleast) in H'; unfold_rels;
      congruence
    end.
  - intros s Hs Hhs. apply HxP; [ | exact Hhs ].
    apply (LP.least_derivable is_local F T Hleast (Atom_sources w s)) in Hs.
    inversion Hs; subst; unfold_rels;
      try solve
        [ exfalso; eapply (ao_constructor _ _ _ Hao); eassumption
        | exfalso; eapply (ao_parameter_base _ _ _ Hao); eassumption ].
    match goal with
    | H : derivable _ _ _ (Atom_alias _ ?from),
      H' : derivable _ _ _ (Atom_sources ?from _) |- _ =>
      rewrite (alias_only_alias w x Hao from H) in H';
      exact (LP.least_holds is_local F T Hleast _ H')
    end.
Qed.

(* ======================================================================== *)
(* The chain of witnesses                                                   *)
(* ======================================================================== *)

(* What [never_returns_of_call] asks once it has the sources [Pw] of a
   witness, with the sizes [nums] of the groups after it. *)
Definition after_witness (Pw : node -> bool) (nums : list nat) (nres : nat)
    : bool :=
  match nums with
  | [] => existsb (fun i => negb (result_has_source_of F S Pw i)) (seq 0 nres)
  | _ :: _ =>
    never_returns_of_call F S Unknown_arity_code_pointer
      (get_field_sources F S Pw first_result) nums nres
  end.

Lemma never_returns_of_call_cons (P : node -> bool) (k : nat)
    (nums : list nat) (nres : nat) :
  never_returns_of_call F S Unknown_arity_code_pointer (Sources P) (k :: nums)
    nres
  = match get_field_sources F S P unknown with
    | Any_source => false
    | Sources Pw => after_witness Pw nums nres
    end.
Proof.
  cbn [never_returns_of_call].
  destruct (get_field_sources F S P unknown); [ reflexivity | ].
  destruct nums; reflexivity.
Qed.

(* Along a chain of witnesses whose first is covered by [Pw], the last
   is covered by the set that [never_returns_of_call] reaches, so a
   result of the rank it finds has no source. *)
Lemma chain_dead (nres : nat) (nums : list nat) :
  check_unknown_chains F = true -> is_local unknown = false ->
  is_local first_result = false ->
  ∀ Pw : node -> bool,
    after_witness Pw nums nres = true ->
    ∃ i : nat,
      i < nres
      ∧ ∀ (g : list simple) (gs : list (list simple)) (w : node)
          (ws returns : list node) (exn called wl W R : node),
          covers Pw w -> length gs = length nums ->
          length ws = length nums ->
          caller_chain_edges G returns exn called ((g, w) :: combine gs ws) ->
          nth_error (w :: ws) (length nums) = Some wl ->
          may_come_from S W wl ->
          g_constructor G W (Field_return_of_call (Normal i)) R ->
          s_has_source S R = false.
Proof.
  pose proof (LP.least_closed is_local F T Hleast) as Hc.
  intros Hcheck Hnu Hnf.
  induction nums as [ | k nums IH]; intros Pw Hafter.
  - unfold after_witness in Hafter.
    apply existsb_exists in Hafter as (i & Hi & Hno). apply in_seq in Hi.
    exists i. split; [ lia | ].
    intros g gs w ws returns exn called wl W R [Hwany HwP] _ Hlen _ Hwl HW
      HR.
    destruct ws; [ | discriminate Hlen ]. cbn in Hwl. injection Hwl as <-.
    destruct HW as (_ & [Hany | HsW] & _); [ congruence | ].
    destruct (s_has_source S R) eqn:ER; [ exfalso | reflexivity ].
    pose proof (has_source_constructor _ _ _ Hc R _ W ER HR) as HhW.
    pose proof (HwP W HsW HhW) as HPW.
    apply negb_true_iff, Bool.not_true_iff_false in Hno. apply Hno.
    unfold result_has_source_of. apply existsb_exists.
    exists (W, Field_return_of_call (Normal i), R). split; [ exact HR | ].
    rewrite HPW, field_eqb_refl, ER. reflexivity.
  - unfold after_witness in Hafter.
    destruct (get_field_sources F S Pw first_result) as [ | Q] eqn:EQ;
      [ discriminate Hafter | ].
    rewrite never_returns_of_call_cons in Hafter.
    destruct (get_field_sources F S Q unknown) as [ | Pw'] eqn:EPw';
      [ discriminate Hafter | ].
    destruct (IH Pw' Hafter) as (i & Hi & Hdead).
    exists i. split; [ exact Hi | ].
    intros g gs w ws returns exn called wl W R Hw Hglen Hlen Hch Hwl HW HR.
    destruct gs as [ | g1 gs]; [ discriminate Hglen | ].
    destruct ws as [ | w1 ws]; [ discriminate Hlen | ].
    cbn [combine caller_chain_edges] in Hch.
    destruct Hch as (_ & _ & _ & (v & Hv & Hw1) & Hch).
    pose proof Hch as Hch1. cbn [caller_chain_edges] in Hch1.
    destruct Hch1 as (_ & _ & Hcalled & _).
    unfold_rels.
    destruct (check_unknown_chains_at F w1 v Hcheck Hw1)
      as (Hro1 & Hwit & _).
    destruct (Hwit (ex_intro _ called Hcalled)) as (w' & _ & Hrov).
    destruct (LP.ro_accessor _ _ _ _ _ Hrov _ _ Hv) as [_ Ew]. subst w'.
    pose proof (read_covers v w first_result Pw Q Hrov Hnf Hw EQ) as Hcv.
    pose proof (read_covers w1 v unknown Q Pw' Hro1 Hnu Hcv EPw') as Hcw1.
    apply (Hdead g1 gs w1 ws returns exn called wl W R Hcw1);
      [ cbn in Hglen; lia | cbn in Hlen; lia | exact Hch | exact Hwl
      | exact HW | exact HR ].
Qed.

(* A call of unknown arity that [apply_never_returns] makes one that
   never returns has a rank of its results that no chain of witnesses it
   reads off its callee gives a source. *)
Lemma never_returns_dead (cds : finite_map code_dep)
    (codes : finite_map code_metadata) (ap : apply_expr) :
  is_local_slots_only is_local -> check_unknown_chains F = true ->
  ap_call_kind ap = Indirect_unknown_arity ->
  apply_never_returns F S cds codes ap = true ->
  ∃ i : nat,
    i < length (unarize (ap_return_arity ap)) ∧ call_dead_rank G S ap i.
Proof.
  intros Hlocal Hcheck Hkind Hnr.
  assert (Hnu : is_local unknown = false)
    by (apply (not_local_field is_local _ Hlocal); intros; discriminate).
  assert (Hnf : is_local first_result = false)
    by (apply (not_local_field is_local _ Hlocal); intros; discriminate).
  set (nres := length (unarize (ap_return_arity ap))) in *.
  set (callee := node_of_simple (ap_callee ap)).
  assert (Hn : never_returns_of_call F S Unknown_arity_code_pointer
                 (get_direct_sources S callee)
                 (map (fun c => length (unarize_component c))
                    (ap_args_arity ap))
                 nres
               = true).
  { unfold apply_never_returns, code_id_actually_called in Hnr.
    rewrite Hkind in Hnr. unfold callee.
    destruct (ap_callee ap); [ exact Hnr | exact Hnr | discriminate Hnr ]. }
  remember (map (fun c => length (unarize_component c)) (ap_args_arity ap))
    as nums eqn:Enums.
  destruct nums as [ | k nums]; [ discriminate Hn | ].
  unfold get_direct_sources in Hn.
  destruct (s_any_source S callee) eqn:Hcany; [ discriminate Hn | ].
  rewrite never_returns_of_call_cons in Hn.
  set (P0 := fun y => s_sources S callee y && s_has_source S y) in Hn.
  destruct (get_field_sources F S P0 unknown) as [ | Pw] eqn:EPw;
    [ discriminate Hn | ].
  destruct (chain_dead nres nums Hcheck Hnu Hnf Pw Hn) as (i & Hi & Hdead).
  exists i. split; [ exact Hi | ].
  intros ctx ws returns exn called apply_node w W R Hlen Hch Hcond Happly Hw
    HW HR.
  assert (Hlg : length (call_groups ap) = length (k :: nums)).
  { rewrite FlambdaReaperWrapperProofs.unknown_groups_length.
    - rewrite <- (length_map (fun c => length (unarize_component c))),
        <- Enums.
      reflexivity.
    - intros E. rewrite E in Enums. discriminate Enums. }
  destruct (call_groups ap) as [ | g gs]; [ discriminate Hlg | ].
  destruct ws as [ | w0 ws]; [ discriminate Hlen | ].
  cbn [combine] in Hch.
  unfold_rels.
  destruct (check_unknown_chains_at F apply_node _ Hcheck Happly)
    as (Hro & _ & Halias & Hprop).
  assert (Hcov : covers P0 (node_of_simple (ap_callee ap))).
  { split; [ exact Hcany | ]. intros s Hs Hhs.
    exact (andb_true_intro (conj Hs Hhs)). }
  pose proof (read_covers apply_node _ unknown P0 Pw Hro Hnu Hcov EPw)
    as Hcov1.
  assert (Hao : alias_only F w0 apply_node).
  { specialize (Hcond w0 eq_refl). unfold cond_alias_edge in Hcond.
    destruct ctx as [id | ]; unfold_rels;
      [ exact (Hprop _ _ Hcond) | exact (Halias _ Hcond) ]. }
  apply (Hdead g gs w0 ws returns exn called w W R
           (alias_covers w0 apply_node Pw Hao Hcov1));
    [ cbn in Hlg; lia | cbn in Hlen, Hlg; lia | exact Hch | | exact HW
    | exact HR ].
  cbn [length pred] in Hw. cbn in Hlen, Hlg.
  replace (length nums) with (length ws) by lia. exact Hw.
Qed.

End Chains.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Theorem indirect_dead_frame : indirect_dead_frame_statement.
Proof.
  intros flag is_local u e T n s ap h vars0 K0 rec0 Hlocal Hslots Hfrag
    Hleast Hcheck Hrun Hexpr Hkind Hexn_ne Hnr Hres.
  destruct (unit_inv flag is_local u e T n s Hlocal Hslots Hfrag
              (proj1 Hleast) Hrun)
    as (noted & deps & Hc & Hconsts & Hsyms & Hunit_syms & Hdeps & Hnd & Hcds
        & Hlrs & Hs).
  destruct (never_returns_dead is_local _ T Hleast _ _ ap Hlocal Hcheck Hkind
              Hnr)
    as (i & Hi & Hdead).
  exists i. split; [ exact Hi | ].
  intros x m s' Hx Hrun'.
  exact (marked_indirect_run_at is_local _ _ noted deps _ _ _ _ u Hc Hlocal
           Hconsts Hsyms Hunit_syms Hdeps Hnd Hcds Hlrs s ap i h vars0 K0 rec0
           x m s' Hs Hexpr Hkind Hexn_ne Hdead Hres Hx Hrun').
Qed.

Theorem chains_example : chains_example_statement.
Proof.
  cbv zeta. split.
  - apply (LP.compute_least_sound _ _ 40). vm_compute. reflexivity.
  - repeat split; vm_compute; reflexivity.
Qed.

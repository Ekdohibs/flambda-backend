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


(* Toward [unused_code_dead] (FlambdaReaperFixFailsSpec.v): the
   [my_closure] of code that is not used has no source in a least
   solution, given the shape of the facts into its node, into the node
   [cd_my_closure] of its code dep and into the nodes that the direct
   calls of the code make with [Auto] ([add_apply]). Nothing here needs
   review.

   - [has_source_into_only]: the inversion of a least solution at a node
     into which the facts only alias ([into_only]).
   - [unused_my_closure_dead]: where the facts have the shape that the
     traversal gives ([shape]), the node of [my_closure] of code whose code
     id has no usage has no source. Its node has a source only through
     [cd_my_closure], which has one only through a closure of the code,
     a propagate on the code id, or a direct call made where its caller
     is used, from the node [callee_if_any_source] of the call, which has
     a source only where the callee has any source. Then the call aliases
     the known witness of the code into its widget, and that into the
     witness whose code id it reads into a node with any usage: the code
     id gets any usage (a node of the chain has any usage) or is read
     into that node through [usages] (none has).

   - [unused_code_dead_of_shapes]: [unused_code_dead] where every code of
     the unit has the shape.

   That the traversal of a unit whose variables are bound once gives the
   facts this shape is not proved (see S7.md). *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaReaperGraphSpec
  FlambdaReaperLeastSpec.
From OxCamlFlambda2 Require FlambdaReaperLeastProofs FlambdaReaperFlowSpec
  FlambdaReaperRebuild FlambdaReaperFixFailsSpec.
Import ListNotations.

Ltac unfold_rels :=
  cbn [graph_of_facts solution_of_tables holds g_alias g_use g_accessor
       g_constructor g_argument g_parameter g_propagate
       g_alias_if_any_source g_any_usage g_any_source g_any_source_if_used
       g_zero_alloc_source s_alias s_any_usage s_any_source s_has_usage
       s_has_source s_usages s_sources s_reading_field s_escaping_field
       s_field_used] in *.

(* ======================================================================== *)
(* A node into which the facts only alias                                   *)
(* ======================================================================== *)

(* The facts put something into [x] only by aliases: from a node of [A],
   or under a propagate or an alias on any source of a pair of [P] or
   [Q] (the node of the condition, the node aliased). *)
Record into_only (F : facts) (x : node) (A : node -> Prop)
    (P Q : node -> node -> Prop) : Prop := {
  io_alias : ∀ from, (x, from) ∈ f_alias F -> A from;
  io_propagate : ∀ c from, (c, x, from) ∈ f_propagate F -> P c from;
  io_if_any_source : ∀ c from,
    (c, x, from) ∈ f_alias_if_any_source F -> Q c from;
  io_accessor : ∀ f b, (x, f, b) ∉ f_accessor F;
  io_parameter_to : ∀ b i, (b, i, x) ∉ f_parameter F;
  io_parameter_base : ∀ i t, (x, i, t) ∉ f_parameter F;
  io_constructor : ∀ r from, (x, r, from) ∉ f_constructor F;
  io_any_source : x ∉ f_any_source F;
  io_zero_alloc : x ∉ f_zero_alloc_source F;
  io_if_used : ∀ c, (c, x) ∉ f_any_source_if_used F;
  io_use : ∀ from, (x, from) ∉ f_use F
}.

Section Into.

Variable is_local : field -> bool.
Variable F : facts.
Variable S : solution.

Abbreviation G := (graph_of_facts F).
Abbreviation D := (derivable is_local G S).

Variable x : node.
Variable A : node -> Prop.
Variables P Q : node -> node -> Prop.
Hypothesis Hio : into_only F x A P Q.

Lemma alias_into_only (from : node) :
  D (Atom_alias x from) ->
  A from ∨ (∃ c, P c from ∧ D (Atom_any_usage c))
  ∨ (∃ c, Q c from ∧ D (Atom_any_source c)).
Proof.
  intros Hd. inversion Hd; subst; unfold_rels;
    try solve [ exfalso; eapply (io_accessor _ _ _ _ _ Hio); eassumption
              | exfalso; eapply (io_parameter_to _ _ _ _ _ Hio);
                eassumption ].
  - left. eapply (io_alias _ _ _ _ _ Hio). eassumption.
  - right; left. exists if_used. split; [ | assumption ].
    eapply (io_propagate _ _ _ _ _ Hio). eassumption.
  - right; right. exists if_any_source. split; [ | assumption ].
    eapply (io_if_any_source _ _ _ _ _ Hio). eassumption.
  - match goal with
    | H : derivable _ _ _ (Atom_reading_field _ _) |- _ =>
        inversion H; subst; unfold_rels
    end.
    exfalso. eapply (io_accessor _ _ _ _ _ Hio). eassumption.
Qed.

Lemma any_source_into_only :
  D (Atom_any_source x) ->
  ∃ from, D (Atom_alias x from) ∧ D (Atom_any_source from).
Proof.
  intros Hd. inversion Hd; subst; unfold_rels.
  - exfalso. eapply (io_any_source _ _ _ _ _ Hio). eassumption.
  - exfalso. eapply (io_zero_alloc _ _ _ _ _ Hio). eassumption.
  - exfalso. eapply (io_if_used _ _ _ _ _ Hio). eassumption.
  - eauto.
  - exfalso. eapply (io_parameter_to _ _ _ _ _ Hio). eassumption.
  - exfalso. eapply (io_accessor _ _ _ _ _ Hio). eassumption.
  - exfalso. eapply (io_use _ _ _ _ _ Hio). eassumption.
Qed.

(* [x] has a source only through an alias. *)
Lemma has_source_into_only :
  D (Atom_has_source x) ->
  ∃ from, D (Atom_alias x from) ∧ D (Atom_has_source from).
Proof.
  intros Hd. inversion Hd; subst; unfold_rels.
  - match goal with
    | H : derivable _ _ _ (Atom_any_source x) |- _ =>
        destruct (any_source_into_only H) as (from & H1 & H2)
    end.
    exists from. split; [ exact H1 | apply by_has_source_any; exact H2 ].
  - eauto.
  - exfalso. eapply (io_constructor _ _ _ _ _ Hio). eassumption.
  - exfalso. eapply (io_constructor _ _ _ _ _ Hio). eassumption.
  - exfalso. eapply (io_parameter_base _ _ _ _ _ Hio). eassumption.
Qed.

(* Both together: the source comes through one of the three ways. *)
Lemma has_source_from (Hd : D (Atom_has_source x)) :
  ∃ from, D (Atom_has_source from)
    ∧ (A from ∨ (∃ c, P c from ∧ D (Atom_any_usage c))
       ∨ (∃ c, Q c from ∧ D (Atom_any_source c))).
Proof.
  destruct (has_source_into_only Hd) as (from & H1 & H2).
  exists from. split; [ exact H2 | exact (alias_into_only from H1) ].
Qed.

End Into.

(* ======================================================================== *)
(* The shape of the facts of a code that is not used                        *)
(* ======================================================================== *)

(* A direct call of the code, with [Auto]: the code id of the code that
   makes it, if any; its callee; the node [callee_if_any_source] that it
   gives the code dep for [my_closure]; its widget and the node
   [widget_if_any_source] ([add_apply]); the witness of the widget and
   the node that its code id is read into
   ([make_known_arity_apply_widget]). *)
Record direct_call : Type := Direct_call {
  dc_caller : option code_id;
  dc_callee : node;
  dc_closure : node;
  dc_widget : node;
  dc_widget_if : node;
  dc_witness : node;
  dc_called : node
}.

(* The alias of [from] into [to_] that a fact of the code [caller] makes
   ([cond_alias], [alias_for_caller]). *)
Definition caller_alias (F : facts) (caller : option code_id)
    (to_ from : node) : Prop :=
  match caller with
  | None => (to_, from) ∈ f_alias F
  | Some c => (Node_code_id c, to_, from) ∈ f_propagate F
  end.

(* The facts around the code [id], whose [my_closure] is [mc], whose code
   dep has [my_closure] the node [cdm] and the known witness [kw], with
   the direct calls [calls]. *)
Record shape (F : facts) (id : code_id) (mc : variable) (cdm kw : node)
    (calls : direct_call -> Prop) : Prop := {
  sh_my_closure :
    into_only F (Node_variable mc) (fun from => from = cdm)
      (fun _ _ => False) (fun _ _ => False);
  sh_code_dep :
    into_only F cdm
      (fun from => ∃ d, calls d ∧ dc_caller d = None ∧ from = dc_closure d)
      (fun c from =>
         c = Node_code_id id
         ∨ ∃ d c', calls d ∧ dc_caller d = Some c' ∧ c = Node_code_id c'
                  ∧ from = dc_closure d)
      (fun _ _ => False);
  sh_closure : ∀ d, calls d ->
    into_only F (dc_closure d) (fun _ => False) (fun _ _ => False)
      (fun c from => c = dc_callee d ∧ from = dc_callee d);
  sh_known : (kw, Field_code_id_of_call_witness, Node_code_id id)
    ∈ f_constructor F;
  sh_widget_if : ∀ d, calls d ->
    caller_alias F (dc_caller d) (dc_widget_if d) kw;
  sh_widget : ∀ d, calls d ->
    (dc_callee d, dc_widget d, dc_widget_if d) ∈ f_alias_if_any_source F;
  sh_witness : ∀ d, calls d ->
    caller_alias F (dc_caller d) (dc_witness d) (dc_widget d);
  sh_read : ∀ d, calls d ->
    (dc_called d, Field_code_id_of_call_witness, dc_witness d)
    ∈ f_accessor F;
  sh_called : ∀ d, calls d -> dc_called d ∈ f_any_usage F
}.

Section Unused.

Variable is_local : field -> bool.
Variable F : facts.
Variable T : tables.
Hypothesis Hleast : least_solution is_local F T.
Hypothesis Hslots : is_local_slots_only is_local.

Abbreviation G := (graph_of_facts F).
Abbreviation S := (solution_of_tables T).
Abbreviation D := (derivable is_local G S).

Lemma closedS : closed is_local G S.
Proof. exact (FlambdaReaperLeastProofs.least_closed is_local F T Hleast). Qed.

Lemma holdsS (a : atom) : D a -> holds S a = true.
Proof. exact (FlambdaReaperLeastProofs.least_holds is_local F T Hleast a). Qed.

Lemma cid_not_local : is_local Field_code_id_of_call_witness = false.
Proof.
  destruct (is_local Field_code_id_of_call_witness) eqn:E; [ | reflexivity ].
  destruct (Hslots _ E) as [(w & H) | (f & H)]; discriminate H.
Qed.

(* An alias made by code that is used, or by the unit. *)
Lemma caller_alias_holds (caller : option code_id) (to_ from : node) :
  caller_alias F caller to_ from ->
  (∀ c, caller = Some c -> s_any_usage S (Node_code_id c) = true) ->
  s_alias S to_ from = true.
Proof.
  intros H Hu. pose proof closedS as Hc. destruct caller as [c | ].
  - exact (alias_propagate _ _ _ Hc _ _ _ (Hu c eq_refl) H).
  - exact (alias_input _ _ _ Hc _ _ H).
Qed.

(* A direct call made, of a callee with any source, makes the code
   used. *)
Lemma call_uses (id : code_id) (mc : variable) (cdm kw : node)
    (calls : direct_call -> Prop) (d : direct_call) :
  shape F id mc cdm kw calls -> calls d ->
  (∀ c, dc_caller d = Some c -> s_any_usage S (Node_code_id c) = true) ->
  s_any_source S (dc_callee d) = true ->
  s_has_usage S (Node_code_id id) = true.
Proof.
  intros Hsh Hd Hcaller Hcallee. pose proof closedS as Hc.
  pose proof (caller_alias_holds _ _ _ (sh_widget_if _ _ _ _ _ _ Hsh d Hd)
                Hcaller) as A1.
  pose proof (alias_if_any_source _ _ _ Hc _ _ _ Hcallee
                (sh_widget _ _ _ _ _ _ Hsh d Hd)) as A2.
  pose proof (caller_alias_holds _ _ _ (sh_witness _ _ _ _ _ _ Hsh d Hd)
                Hcaller) as A3.
  pose proof (sh_read _ _ _ _ _ _ Hsh d Hd) as Hread.
  pose proof (has_usage_any _ _ _ Hc _
                (any_usage_input _ _ _ Hc _ (sh_called _ _ _ _ _ _ Hsh d Hd)))
    as Hcalled.
  pose proof (sh_known _ _ _ _ _ _ Hsh) as Hkw.
  destruct (s_any_usage S kw) eqn:Ekw.
  - apply (has_usage_any _ _ _ Hc).
    exact (any_usage_constructor _ _ _ Hc _ _ _ Ekw Hkw cid_not_local).
  - assert (Ewif : s_any_usage S (dc_widget_if d) = false).
    { destruct (s_any_usage S (dc_widget_if d)) eqn:E; [ | reflexivity ].
      rewrite (any_usage_alias _ _ _ Hc _ _ E A1) in Ekw. discriminate. }
    assert (Ew : s_any_usage S (dc_widget d) = false).
    { destruct (s_any_usage S (dc_widget d)) eqn:E; [ | reflexivity ].
      rewrite (any_usage_alias _ _ _ Hc _ _ E A2) in Ewif. discriminate. }
    assert (Ewit : s_any_usage S (dc_witness d) = false).
    { destruct (s_any_usage S (dc_witness d)) eqn:E; [ | reflexivity ].
      rewrite (any_usage_alias _ _ _ Hc _ _ E A3) in Ew. discriminate. }
    pose proof (usages_accessor _ _ _ Hc _ _ _ Hread Ewit) as U0.
    pose proof (usages_alias _ _ _ Hc _ _ _ U0 Ewit A3 Ew) as U1.
    pose proof (usages_alias _ _ _ Hc _ _ _ U1 Ew A2 Ewif) as U2.
    pose proof (usages_alias _ _ _ Hc _ _ _ U2 Ewif A1 Ekw) as U3.
    pose proof (alias_constructor_usage _ _ _ Hc _ _ _ _ _ Hkw U3 Ekw Hread)
      as A4.
    exact (has_usage_alias _ _ _ Hc _ _ Hcalled A4).
Qed.

(* The node of [my_closure] of code whose code id has no usage has no
   source. *)
Lemma unused_my_closure_dead (id : code_id) (mc : variable) (cdm kw : node)
    (calls : direct_call -> Prop) :
  shape F id mc cdm kw calls ->
  s_has_usage S (Node_code_id id) = false ->
  s_has_source S (Node_variable mc) = false.
Proof.
  intros Hsh Hun. pose proof closedS as Hc.
  destruct (s_has_source S (Node_variable mc)) eqn:E; [ | reflexivity ].
  exfalso.
  pose proof (FlambdaReaperLeastProofs.least_derivable is_local F T Hleast
                (Atom_has_source (Node_variable mc)) E) as Hd.
  destruct (has_source_from is_local F S _ _ _ _
              (sh_my_closure _ _ _ _ _ _ Hsh) Hd)
    as (from & Hfrom & [-> | [(c & [] & _) | (c & [] & _)]]).
  destruct (has_source_from is_local F S _ _ _ _
              (sh_code_dep _ _ _ _ _ _ Hsh) Hfrom)
    as (cl & Hcl & [(d & Hd' & Hnone & ->)
                   | [(c & [-> | (d & c' & Hd' & Hsome & -> & ->)] & Hu)
                     | (c & [] & _)]]).
  - destruct (has_source_from is_local F S _ _ _ _
                (sh_closure _ _ _ _ _ _ Hsh d Hd') Hcl)
      as (cal & _ & [[] | [(c & [] & _) | (c & [-> ->] & Hs)]]).
    apply holdsS in Hs. cbn [holds] in Hs.
    assert (Hcaller : ∀ c, dc_caller d = Some c ->
                        s_any_usage S (Node_code_id c) = true)
      by (intros c E'; rewrite Hnone in E'; discriminate E').
    rewrite (call_uses id mc cdm kw calls d Hsh Hd' Hcaller Hs) in Hun.
    discriminate.
  - apply holdsS in Hu. cbn [holds] in Hu.
    rewrite (has_usage_any _ _ _ Hc _ Hu) in Hun. discriminate.
  - destruct (has_source_from is_local F S _ _ _ _
                (sh_closure _ _ _ _ _ _ Hsh d Hd') Hcl)
      as (cal & _ & [[] | [(c & [] & _) | (c & [-> ->] & Hs)]]).
    apply holdsS in Hs. cbn [holds] in Hs. apply holdsS in Hu.
    cbn [holds] in Hu.
    assert (Hcaller : ∀ c, dc_caller d = Some c ->
                        s_any_usage S (Node_code_id c) = true).
    { intros c E'. rewrite Hsome in E'. injection E' as <-. exact Hu. }
    rewrite (call_uses id mc cdm kw calls d Hsh Hd' Hcaller Hs) in Hun.
    discriminate.
Qed.

(* [unused_code_dead] holds where the facts of each code of the unit have
   the shape. *)
Lemma unused_code_dead_of_shapes (e : expr) :
  (∀ id c, (id, c) ∈ FlambdaReaperFlowSpec.unit_codes e ->
     ∃ cdm kw calls, shape F id (code_my_closure c) cdm kw calls) ->
  FlambdaReaperFixFailsSpec.unused_code_dead S e = true.
Proof.
  intros H. unfold FlambdaReaperFixFailsSpec.unused_code_dead.
  apply forallb_forall. intros [id c] Hc.
  destruct (FlambdaReaperRebuild.is_code_id_used S id) eqn:Eu;
    [ reflexivity | ].
  destruct (H id c Hc) as (cdm & kw & calls & Hsh).
  cbn [orb]. apply negb_true_iff.
  exact (unused_my_closure_dead id (code_my_closure c) cdm kw calls Hsh Eu).
Qed.

End Unused.

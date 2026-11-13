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

(* The proofs of FlambdaReaperTypesSpec.v. Nothing here needs review.

   The queries of the solution are read back as the relations of the
   graph and of the solution (the first section); [follow_field] and
   [metadata_for_name] keep [metadata_at] (the second); the rewriting is
   then proved sound by induction on its fuel, [rewrite_arbitrary] and
   [rewrite] together ([rewrite_sound]), with the lemma that a metadata
   that forgets gives a type that every value of the kind has
   ([rewrite_forgets]). The closure case ([closure_sound]) reads the
   closure off the head of its type ([meet_closures_head],
   [closure_of_type]) and off the rebuild ([closure_rebuilt_view]); the
   set of the closure is among those that [refine_sets_of_closures]
   keeps ([refine_keeps]: in a least solution an allocation is a source
   of each usage of it that has not any source,
   [usages_carry_sources]); the metadata of its value slots holds where
   they are ([follow_field_for_set_at]), and the join of the slots of
   the sets kept describes what that of the set does
   ([fold_join_covered]). The counterexamples are computed. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaPrimitiveSpec
  FlambdaPrimitiveProofs FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperRebuild
  FlambdaReaperSolution
  FlambdaReaperSolutionSpec FlambdaReaperSolutionProofs
  FlambdaReaperFlowSpec FlambdaReaperLeast FlambdaReaperLeastSpec
  FlambdaReaperLeastProofs FlambdaReaperTypes FlambdaReaperTypesSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeMeaningSpec FlambdaTypeMeaningProofs
  FlambdaBindingTime FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaTypingEnvProofs FlambdaExpandHead FlambdaExpandHeadSpec
  FlambdaExpandHeadProofs FlambdaTypeAccessSpec FlambdaProvers
  FlambdaTypeMeaningBoolSpec FlambdaTypeMeaningBoolProofs
  FlambdaEnvAliases FlambdaEnvAliasesSpec FlambdaEnvAliasesProofs.
Import ListNotations.

(* ======================================================================== *)
(* Sets of nodes                                                            *)
(* ======================================================================== *)

Lemma node_eqb_decides : decides_equality node_eqb.
Proof.
  intros n n'. split; [ apply node_eqb_eq | intros ->; apply node_eqb_refl ].
Qed.

Lemma In_dedup_nodes (x : node) (l : list node) :
  x ∈ dedup_nodes l <-> x ∈ l.
Proof.
  unfold dedup_nodes. rewrite <- !mem_node_In.
  pose proof (keyed_among_dedup node node_eqb node_eqb_decides x l) as H.
  unfold Keyed.among in H. unfold mem_node. rewrite H. reflexivity.
Qed.

Lemma In_filter_map_snd (x y : node) (l : list (node * node)) :
  y ∈ map snd (filter (fun p => node_eqb (fst p) x) l) <-> (x, y) ∈ l.
Proof.
  rewrite in_map_iff. split.
  - intros ([x' y'] & Hy & Hin). simpl in Hy. subst y'.
    apply filter_In in Hin as [Hin Hx]. simpl in Hx.
    apply node_eqb_eq in Hx. subst x'. exact Hin.
  - intros Hin. exists (x, y). split; [ reflexivity | ].
    apply filter_In. split; [ exact Hin | apply node_eqb_refl ].
Qed.

Lemma In_filter_map_fst (x y : node) (l : list (node * node)) :
  x ∈ map fst (filter (fun p => node_eqb (snd p) y) l) <-> (x, y) ∈ l.
Proof.
  rewrite in_map_iff. split.
  - intros ([x' y'] & Hx & Hin). simpl in Hx. subst x'.
    apply filter_In in Hin as [Hin Hy]. simpl in Hy.
    apply node_eqb_eq in Hy. subst y'. exact Hin.
  - intros Hin. exists (x, y). split; [ reflexivity | ].
    apply filter_In. split; [ exact Hin | apply node_eqb_refl ].
Qed.

(* ======================================================================== *)
(* The queries, read back                                                   *)
(* ======================================================================== *)

Section Queries.

Variable ctx : rewrite_context.

Local Abbreviation G := (graph_of_facts (rc_facts ctx)).
Local Abbreviation sol := (solution_of_tables (rc_tables ctx)).

Lemma usages_of_iff (x y : node) :
  y ∈ usages_of ctx x <-> s_usages sol x y = true.
Proof.
  unfold usages_of. rewrite In_filter_map_snd. cbn [solution_of_tables
    s_usages]. rewrite mem_pair_In. reflexivity.
Qed.

Lemma sources_of_iff (x y : node) :
  y ∈ sources_of ctx x <-> s_sources sol x y = true.
Proof.
  unfold sources_of. rewrite In_filter_map_snd. cbn [solution_of_tables
    s_sources]. rewrite mem_pair_In. reflexivity.
Qed.

Lemma constructors_of_iff (a from : node) (f : field) :
  from ∈ constructors_of ctx a f <-> g_constructor G a f from.
Proof.
  unfold constructors_of. cbn [graph_of_facts g_constructor].
  rewrite in_map_iff. split.
  - intros ([[b f'] from'] & Hfrom & Hin). simpl in Hfrom. subst from'.
    apply filter_In in Hin as [Hin Hb]. apply andb_true_iff in Hb as [Hb Hf].
    apply node_eqb_eq in Hb. apply field_eqb_eq in Hf. subst. exact Hin.
  - intros Hin. exists (a, f, from). split; [ reflexivity | ].
    apply filter_In. rewrite node_eqb_refl, field_eqb_refl. auto.
Qed.

Lemma accessors_of_iff (f : field) (base to_ : node) :
  to_ ∈ accessors_of ctx f base <-> g_accessor G to_ f base.
Proof.
  unfold accessors_of. cbn [graph_of_facts g_accessor].
  rewrite in_map_iff. split.
  - intros ([[to' f'] b] & Hto & Hin). simpl in Hto. subst to'.
    apply filter_In in Hin as [Hin Hb]. apply andb_true_iff in Hb as [Hf Hb].
    apply node_eqb_eq in Hb. apply field_eqb_eq in Hf. subst. exact Hin.
  - intros Hin. exists (to_, f, base). split; [ reflexivity | ].
    apply filter_In. rewrite node_eqb_refl, field_eqb_refl. auto.
Qed.

Lemma In_filter_has_use (x : node) (l : list node) :
  x ∈ filter (has_use ctx) l <-> x ∈ l ∧ s_has_usage sol x = true.
Proof. apply filter_In. Qed.

Lemma In_filter_has_source (x : node) (l : list node) :
  x ∈ filter (has_source ctx) l <-> x ∈ l ∧ s_has_source sol x = true.
Proof. apply filter_In. Qed.

(* [get_direct_usages]. *)
Lemma get_direct_usages_iff (xs : list node) (z : node) :
  z ∈ get_direct_usages ctx xs
  <-> ∃ x : node,
        x ∈ xs ∧ s_usages sol x z = true ∧ s_has_usage sol z = true.
Proof.
  unfold get_direct_usages. rewrite In_dedup_nodes, in_flat_map.
  split.
  - intros (x & Hx & Hz). apply In_filter_has_use in Hz as [Hz Hhu].
    apply usages_of_iff in Hz. exists x. auto.
  - intros (x & Hx & Hz & Hhu). exists x. split; [ exact Hx | ].
    apply In_filter_has_use. split; [ apply usages_of_iff; exact Hz | ].
    exact Hhu.
Qed.

(* [get_usages]. *)
Lemma get_usages_spec (x : node) :
  match get_usages ctx x with
  | Bottom => s_has_usage sol x = false
  | Unknown => s_any_usage sol x = true ∧ s_has_usage sol x = true
  | Ok usages =>
    s_has_usage sol x = true ∧ s_any_usage sol x = false
    ∧ ∀ z : node,
        z ∈ usages <-> s_usages sol x z = true ∧ s_has_usage sol z = true
  end.
Proof.
  unfold get_usages. cbn [solution_of_tables s_has_usage s_any_usage].
  unfold has_use, any_usage.
  destruct (mem_node x (t_has_usage (rc_tables ctx))) eqn:Hhu;
    [ | reflexivity ].
  destruct (mem_node x (t_any_usage (rc_tables ctx))) eqn:Hau;
    simpl; [ auto | ].
  split; [ reflexivity | split; [ reflexivity | ] ].
  intros z. rewrite get_direct_usages_iff. split.
  - intros (x' & [-> | []] & Hz). exact Hz.
  - intros Hz. exists x. split; [ left; reflexivity | exact Hz ].
Qed.

(* [get_single_source]. *)
Lemma get_single_source_spec (x : node) :
  match get_single_source ctx x with
  | Bottom => s_has_source sol x = false
  | Ok y =>
    s_has_source sol x = true ∧ s_any_source sol x = false
    ∧ ∀ y' : node,
        s_sources sol x y' = true -> s_has_source sol y' = true -> y' = y
  | Unknown => True
  end.
Proof.
  unfold get_single_source. cbn [solution_of_tables s_has_source
    s_any_source s_sources].
  unfold has_source, any_source.
  destruct (mem_node x (t_has_source (rc_tables ctx))) eqn:Hhs;
    [ | reflexivity ].
  destruct (mem_node x (t_any_source (rc_tables ctx))) eqn:Has;
    [ exact I | ].
  destruct (allocation_points ctx x) as [ | y [ | y2 rest]] eqn:Hpoints;
    try exact I.
  split; [ reflexivity | split; [ reflexivity | ] ].
  intros y' Hy' Hhs'.
  assert (Hin : y' ∈ allocation_points ctx x).
  { unfold allocation_points. apply In_dedup_nodes, In_filter_has_source.
    split; [ | exact Hhs' ].
    apply sources_of_iff. exact Hy'. }
  rewrite Hpoints in Hin. destruct Hin as [-> | []]. reflexivity.
Qed.

(* [get_single_field_source]. *)
Lemma get_single_field_source_spec (a : node) (f : field) :
  match get_single_field_source ctx a f with
  | Field_no_source =>
    ∀ from : node,
      g_constructor G a f from ->
      s_any_source sol from = false
      ∧ ∀ s : node,
          s_sources sol from s = true -> s_has_source sol s = true -> False
  | Field_one_source s =>
    ∀ from : node,
      g_constructor G a f from ->
      s_any_source sol from = false
      ∧ ∀ s' : node,
          s_sources sol from s' = true -> s_has_source sol s' = true -> s' = s
  | Field_many_sources => True
  end.
Proof.
  unfold get_single_field_source.
  destruct (existsb (any_source ctx) (constructors_of ctx a f)) eqn:Hany;
    [ exact I | ].
  assert (Hnot_any : ∀ from : node,
            g_constructor G a f from -> s_any_source sol from = false).
  { intros from Hfrom. apply constructors_of_iff in Hfrom.
    apply not_true_iff_false. intros Has.
    apply not_true_iff_false in Hany. apply Hany.
    apply existsb_exists. exists from. auto. }
  assert (Hin : ∀ from s : node,
            g_constructor G a f from -> s_sources sol from s = true ->
            s_has_source sol s = true ->
            s ∈ dedup_nodes
                  (flat_map (fun from => filter (has_source ctx)
                                           (sources_of ctx from))
                     (constructors_of ctx a f))).
  { intros from s Hfrom Hs Hhs. apply In_dedup_nodes, in_flat_map.
    exists from. split; [ apply constructors_of_iff; exact Hfrom | ].
    apply In_filter_has_source. split; [ | exact Hhs ].
    apply sources_of_iff. exact Hs. }
  destruct (dedup_nodes
              (flat_map (fun from => filter (has_source ctx)
                                       (sources_of ctx from))
                 (constructors_of ctx a f)))
    as [ | s [ | s2 rest]] eqn:Hl; try exact I.
  - intros from Hfrom. split; [ exact (Hnot_any from Hfrom) | ].
    intros s Hs Hhs. exact (Hin from s Hfrom Hs Hhs).
  - intros from Hfrom. split; [ exact (Hnot_any from Hfrom) | ].
    intros s' Hs Hhs. destruct (Hin from s' Hfrom Hs Hhs) as [-> | []].
    reflexivity.
Qed.

(* [get_one_field_usage]. *)
Lemma get_one_field_usage_spec (f : field) (usages : list node) :
  match get_one_field_usage ctx f usages with
  | Unknown =>
    ∃ u y : node,
      u ∈ usages ∧ g_accessor G y f u ∧ s_any_usage sol y = true
  | Bottom =>
    ∀ u y : node,
      u ∈ usages -> g_accessor G y f u -> s_has_usage sol y = false
  | Ok ys =>
    ys <> []
    ∧ (∀ u y : node,
         u ∈ usages -> g_accessor G y f u -> s_any_usage sol y = false)
    ∧ ∀ y : node,
        y ∈ ys ->
        s_has_usage sol y = true ∧ ∃ u : node, u ∈ usages ∧ g_accessor G y f u
  end.
Proof.
  unfold get_one_field_usage.
  destruct (existsb (any_usage ctx) (flat_map (accessors_of ctx f) usages))
    eqn:Hany.
  { apply existsb_exists in Hany as (y & Hy & Hau).
    apply in_flat_map in Hy as (u & Hu & Hy). apply accessors_of_iff in Hy.
    exists u, y. auto. }
  assert (Hnot_any : ∀ u y : node,
            u ∈ usages -> g_accessor G y f u -> s_any_usage sol y = false).
  { intros u y Hu Hy. apply not_true_iff_false. intros Hau.
    apply not_true_iff_false in Hany. apply Hany. apply existsb_exists.
    exists y. split; [ | exact Hau ]. apply in_flat_map. exists u.
    split; [ exact Hu | apply accessors_of_iff; exact Hy ]. }
  destruct (dedup_nodes (filter (has_use ctx)
                           (flat_map (accessors_of ctx f) usages)))
    as [ | y0 ys] eqn:Hl.
  - intros u y Hu Hy. apply not_true_iff_false. intros Hhu.
    assert (Hin : y ∈ dedup_nodes (filter (has_use ctx)
                                     (flat_map (accessors_of ctx f) usages))).
    { apply In_dedup_nodes, In_filter_has_use. split; [ | exact Hhu ].
      apply in_flat_map. exists u.
      split; [ exact Hu | apply accessors_of_iff; exact Hy ]. }
    rewrite Hl in Hin. destruct Hin.
  - split; [ discriminate | split; [ exact Hnot_any | ] ].
    intros y Hy. rewrite <- Hl in Hy.
    apply In_dedup_nodes, In_filter_has_use in Hy as [Hy Hhu].
    apply in_flat_map in Hy as (u & Hu & Hy). apply accessors_of_iff in Hy.
    split; [ exact Hhu | exists u; auto ].
Qed.

End Queries.

(* A boolean that is both: what [congruence] does, up to conversion. *)
Lemma bool_both (b : bool) : b = true -> b = false -> False.
Proof. intros -> H. discriminate H. Qed.

(* [get_direct_sources] of one node. *)
Lemma get_direct_sources_one (ctx : rewrite_context) (u : node) :
  get_direct_sources ctx [u]
  = if any_source ctx u then Sources_any
    else
      Sources_known (dedup_nodes (filter (has_source ctx) (sources_of ctx u))).
Proof.
  unfold get_direct_sources. cbn [existsb flat_map].
  rewrite orb_false_r, app_nil_r. reflexivity.
Qed.

(* A node of every list of a list of lists is in their intersection. *)
Lemma In_intersect_all (x : node) (ls : list (list node)) :
  ls <> [] -> (∀ l : list node, l ∈ ls -> x ∈ l) -> x ∈ intersect_all ls.
Proof.
  destruct ls as [ | first rest]; [ intros H; contradiction | intros _ Hall ].
  unfold intersect_all.
  assert (Hfold : ∀ (rest : list (list node)) (acc : list node),
             x ∈ acc -> (∀ l : list node, l ∈ rest -> x ∈ l) ->
             x ∈ List.fold_left inter_nodes rest acc).
  { intros rest0. induction rest0 as [ | l rest0 IH]; intros acc Hacc Hl;
      [ exact Hacc | ].
    cbn [List.fold_left].
    apply IH; [ | intros l' Hl'; apply Hl; right; exact Hl' ].
    unfold inter_nodes. apply filter_In. split; [ exact Hacc | ].
    apply mem_node_In. apply Hl. left. reflexivity. }
  apply Hfold; [ apply Hall; left; reflexivity | ].
  intros l Hl. apply Hall. right. exact Hl.
Qed.

(* The lookup of a map of pairs made from a map, keeping the keys. *)
Lemma lookup_map_keys {A B : Type} (h : positive * A -> B) (k : positive)
    (l : list (positive * A)) :
  lookup k (map (fun p => (fst p, h p)) l)
  = match lookup k l with
    | Some a => Some (h (k, a))
    | None => None
    end.
Proof.
  induction l as [ | [k' a'] l IH]; [ reflexivity | ].
  cbn [map fst snd lookup]. destruct (Pos.eqb_spec k k') as [-> | Hne];
    [ reflexivity | exact IH ].
Qed.

Lemma domain_map_keys {A B : Type} (h : positive * A -> B)
    (l : list (positive * A)) :
  domain (map (fun p => (fst p, h p)) l) = domain l.
Proof.
  unfold domain. rewrite map_map. cbn [fst]. reflexivity.
Qed.

(* [Map.inter] of maps keyed by slots, looked up. *)
Lemma lookup_inter_slots {A B C : Type} (f : A -> B -> C)
    (l : finite_map A) (l' : finite_map B) (k : positive) (c : C) :
  lookup k (inter_slots f l l') = Some c ->
  ∃ (a : A) (b : B), lookup k l = Some a ∧ lookup k l' = Some b ∧ c = f a b.
Proof.
  intros H. apply lookup_in in H. unfold inter_slots in H.
  apply in_flat_map in H as ([k0 a] & Hin & Hp). cbn [fst snd] in Hp.
  destruct (lookup k0 l') as [b | ] eqn:Hl'; [ | destruct Hp ].
  destruct Hp as [[= <- <-] | []].
  apply in_compact in Hin. exists a, b. auto.
Qed.

(* ======================================================================== *)
(* What a least solution gives                                              *)
(* ======================================================================== *)

Section Least_facts.

Variable is_local : field -> bool.
Variable F : facts.
Variable T : tables.
Hypothesis Hleast : least_solution is_local F T.

Local Abbreviation G := (graph_of_facts F).
Local Abbreviation sol := (solution_of_tables T).

Lemma least_closed_tables : closed is_local G sol.
Proof. exact (least_closed is_local F T Hleast). Qed.

Lemma usages_not_any (x u : node) :
  s_usages sol x u = true -> s_any_usage sol x = false.
Proof. exact (proj1 (least_no_stale is_local F T x u Hleast)). Qed.

Lemma sources_not_any (x s : node) :
  s_sources sol x s = true -> s_any_source sol x = false.
Proof. exact (proj2 (least_no_stale is_local F T x s Hleast)). Qed.

(* What may come to a node may come to each of its usages, which have
   any source if the node has. *)
Lemma usages_carry_sources (x u : node) :
  s_usages sol x u = true ->
  (∀ s : node,
     s_sources sol x s = true ->
     s_sources sol u s = true ∨ s_any_source sol u = true)
  ∧ (s_any_source sol x = true -> s_any_source sol u = true).
Proof.
  intros H.
  pose proof (least_derivable is_local F T Hleast (Atom_usages x u) H) as Hd.
  clear H. remember (Atom_usages x u) as atom eqn:Ha. revert x u Ha.
  induction Hd; intros x0 u0 Ha; try discriminate Ha.
  - injection Ha as <- <-. split; [ intros s Hs; left; exact Hs | auto ].
  - injection Ha as <- <-. split; [ intros s Hs; left; exact Hs | auto ].
  - injection Ha as <- <-.
    destruct (IHHd1 to_ usage eq_refl) as [IHs IHa].
    pose proof (least_holds is_local F T Hleast _ Hd2) as Halias.
    cbn [holds] in Halias. split.
    + intros s Hs.
      destruct (s_any_source sol to_) eqn:Hat;
        [ right; exact (IHa eq_refl) | ].
      apply IHs.
      exact (sources_alias _ _ _ least_closed_tables from to_ s Hs
               (sources_not_any from s Hs) Halias Hat).
    + intros Has. apply IHa.
      exact (any_source_alias _ _ _ least_closed_tables from to_ Halias Has).
Qed.

(* A node that has a source has any source, or a source that has a
   source. *)
Lemma has_source_sources (x : node) :
  s_has_source sol x = true ->
  s_any_source sol x = true
  ∨ ∃ y : node, s_sources sol x y = true ∧ s_has_source sol y = true.
Proof.
  intros H.
  pose proof (least_derivable is_local F T Hleast (Atom_has_source x) H)
    as Hd.
  clear H. remember (Atom_has_source x) as atom eqn:Ha. revert x Ha.
  induction Hd; intros x0 Ha; try discriminate Ha; injection Ha as <-.
  - left. exact (least_holds is_local F T Hleast _ Hd).
  - destruct (s_any_source sol to_) eqn:Hat; [ left; reflexivity | right ].
    pose proof (least_holds is_local F T Hleast _ Hd2) as Halias.
    cbn [holds] in Halias.
    destruct (IHHd1 from eq_refl) as [Haf | (y & Hy & Hhy)].
    + rewrite (any_source_alias _ _ _ least_closed_tables from to_ Halias Haf)
        in Hat.
      discriminate Hat.
    + exists y. split; [ | exact Hhy ].
      exact (sources_alias _ _ _ least_closed_tables from to_ y Hy
               (sources_not_any from y Hy) Halias Hat).
  - destruct (s_any_source sol base) eqn:Hab; [ left; reflexivity | right ].
    exists base. split.
    + exact (sources_constructor _ _ _ least_closed_tables from relation base
               H Hab).
    + exact (least_holds is_local F T Hleast (Atom_has_source base)
               (by_has_source_constructor _ _ _ from relation base Hd H)).
  - destruct (s_any_source sol base) eqn:Hab; [ left; reflexivity | right ].
    exists base. split.
    + exact (sources_constructor _ _ _ least_closed_tables from relation base
               H Hab).
    + exact (has_source_function_slot _ _ _ least_closed_tables from relation
               base H H0).
  - destruct (s_any_source sol base) eqn:Hab; [ left; reflexivity | right ].
    exists base. split.
    + exact (sources_parameter _ _ _ least_closed_tables to_ relation base H
               Hab).
    + exact (has_source_parameter _ _ _ least_closed_tables to_ relation base
               (least_holds is_local F T Hleast _ Hd) H).
Qed.

Variable sets : list set_of_closures.

Local Abbreviation ctx := (plain_context is_local F T sets).

(* A field of which [field_used_top] holds: what an allocation holds
   there has any usage. *)
Lemma field_used_top_any_usage (c from : node) (f : field) :
  field_used_top ctx c f = true -> g_constructor G c f from ->
  s_any_usage sol from = true.
Proof.
  intros Htop Hcons. pose proof least_closed_tables as Hc.
  unfold field_used_top in Htop. cbn [plain_context rc_is_local] in Htop.
  apply andb_true_iff in Htop as [_ Htop].
  repeat apply orb_true_iff in Htop as [Htop | Htop].
  - apply andb_true_iff in Htop as [Hau Hnl].
    apply negb_true_iff in Hnl.
    exact (any_usage_constructor _ _ _ Hc c f from Hau Hcons Hnl).
  - apply existsb_exists in Htop as (u & Hu & Hy).
    apply existsb_exists in Hy as (y & Hy & Hauy).
    apply usages_of_iff in Hu. apply accessors_of_iff in Hy.
    assert (Halias : s_alias sol y from = true).
    { destruct (is_local f) eqn:Hloc.
      - exact (alias_local_constructor_usage _ _ _ Hc c u f y from Hloc Hcons
                 Hu Hy).
      - exact (alias_constructor_usage _ _ _ Hc c u f y from Hcons Hu
                 (usages_not_any c u Hu) Hy). }
    exact (any_usage_alias _ _ _ Hc y from Hauy Halias).
  - apply andb_true_iff in Htop as [Hau Hread].
    apply existsb_exists in Hread as ([f' y] & Hin & Hy).
    apply andb_true_iff in Hy as [Hf Hauy]. cbn [fst snd] in Hf, Hauy.
    apply field_eqb_eq in Hf. subst f'.
    destruct (is_local f) eqn:Hloc;
      [ | exact (any_usage_constructor _ _ _ Hc c f from Hau Hcons Hloc) ].
    pose proof (escaping_field_rule _ _ _ Hc c f from Hau Hcons Hloc) as Hesc.
    assert (Hrf : s_reading_field sol f y = true).
    { cbn [solution_of_tables s_reading_field].
      apply mem_field_node_In. exact Hin. }
    exact (any_usage_alias _ _ _ Hc y from Hauy
             (alias_escaping_reading _ _ _ Hc f y from Hesc Hrf)).
  - apply andb_true_iff in Htop as [Htop Hsrc].
    apply andb_true_iff in Htop as [Hloc Hau].
    apply existsb_exists in Hsrc as (u & Hu & Hy).
    apply existsb_exists in Hy as (y & Hy & Hauy).
    apply (In_filter_map_fst u c) in Hu.
    apply accessors_of_iff in Hy.
    assert (Hsu : s_sources sol u c = true).
    { cbn [solution_of_tables s_sources]. apply mem_pair_In. exact Hu. }
    exact (any_usage_alias _ _ _ Hc y from Hauy
             (alias_local_accessor_source _ _ _ Hc u c f y from Hloc Hy Hsu
                Hcons)).
Qed.

(* A node of [field_used_as] of an allocation, where [field_used_top] does
   not hold: it has a use, not any usage, and is an alias of what the
   allocation holds there. *)
Lemma field_used_as_alias (c from y : node) (f : field) :
  y ∈ field_used_as ctx c f -> g_constructor G c f from ->
  field_used_top ctx c f = false ->
  s_has_usage sol y = true ∧ s_any_usage sol y = false
  ∧ s_alias sol y from = true.
Proof.
  intros Hy Hcons Htop. pose proof least_closed_tables as Hc.
  assert (Hne : is_nil (constructors_of ctx c f) = false).
  { destruct (constructors_of ctx c f) eqn:Hl; [ | reflexivity ].
    apply (constructors_of_iff ctx) in Hcons. rewrite Hl in Hcons.
    destruct Hcons. }
  unfold field_used_as in Hy. unfold field_used_top in Htop.
  rewrite Hne in Hy, Htop.
  cbn [negb andb plain_context rc_is_local] in Hy, Htop.
  apply (proj1 (In_dedup_nodes y _)) in Hy.
  repeat apply orb_false_iff in Htop as [Htop ?].
  rename H into Hnone4, H0 into Hnone3, H1 into Hnone2.
  apply in_app_or in Hy as [Hy | Hy]; [ | apply in_app_or in Hy as [Hy | Hy] ].
  - apply in_flat_map in Hy as (u & Hu & Hy).
    apply In_filter_has_use in Hy as [Hy Hhu].
    apply usages_of_iff in Hu. apply accessors_of_iff in Hy.
    split; [ exact Hhu | split ].
    + apply not_true_iff_false. intros Hauy.
      apply not_true_iff_false in Hnone2. apply Hnone2.
      apply existsb_exists. exists u.
      split; [ apply usages_of_iff; exact Hu | ].
      apply existsb_exists. exists y.
      split; [ apply accessors_of_iff; exact Hy | exact Hauy ].
    + destruct (is_local f) eqn:Hloc.
      * exact (alias_local_constructor_usage _ _ _ Hc c u f y from Hloc Hcons
                 Hu Hy).
      * exact (alias_constructor_usage _ _ _ Hc c u f y from Hcons Hu
                 (usages_not_any c u Hu) Hy).
  - destruct (any_usage ctx c) eqn:Hau; [ | destruct Hy ].
    apply in_map_iff in Hy as ([f' y'] & Hy' & Hin). cbn [snd] in Hy'.
    subst y'. apply filter_In in Hin as [Hin Hfy].
    apply andb_true_iff in Hfy as [Hf Hhu]. cbn [fst snd] in Hf, Hhu.
    apply field_eqb_eq in Hf. subst f'.
    destruct (is_local f) eqn:Hloc; [ | discriminate Htop ].
    split; [ exact Hhu | split ].
    + apply not_true_iff_false. intros Hauy.
      apply not_true_iff_false in Hnone3. apply Hnone3.
      apply existsb_exists. exists (f, y).
      split; [ exact Hin | cbn; rewrite field_eqb_refl; exact Hauy ].
    + pose proof (escaping_field_rule _ _ _ Hc c f from Hau Hcons Hloc)
        as Hesc.
      assert (Hrf : s_reading_field sol f y = true).
      { cbn [solution_of_tables s_reading_field].
        apply mem_field_node_In. exact Hin. }
      exact (alias_escaping_reading _ _ _ Hc f y from Hesc Hrf).
  - destruct (is_local f && any_usage ctx c) eqn:Hla; [ | destruct Hy ].
    apply andb_true_iff in Hla as [Hloc Hau].
    apply in_flat_map in Hy as (u & Hu & Hy).
    apply In_filter_has_use in Hy as [Hy Hhu].
    apply (In_filter_map_fst u c) in Hu. apply accessors_of_iff in Hy.
    assert (Hsu : s_sources sol u c = true).
    { cbn [solution_of_tables s_sources]. apply mem_pair_In. exact Hu. }
    split; [ exact Hhu | split ].
    + apply not_true_iff_false. intros Hauy.
      apply not_true_iff_false in Hnone4. apply Hnone4.
      cbn [andb]. apply existsb_exists. exists u.
      split; [ apply (In_filter_map_fst u c); exact Hu | ].
      apply existsb_exists. exists y.
      split; [ apply accessors_of_iff; exact Hy | exact Hauy ].
    + exact (alias_local_accessor_source _ _ _ Hc u c f y from Hloc Hy Hsu
               Hcons).
Qed.

(* [get_one_field_usage_of_constructors]. *)
Lemma get_one_field_usage_of_constructors_spec (cs : list node) (f : field) :
  match get_one_field_usage_of_constructors ctx cs f with
  | Unknown => ∃ c : node, c ∈ cs ∧ field_used_top ctx c f = true
  | Bottom => True
  | Ok ys =>
    (∀ c : node, c ∈ cs -> field_used_top ctx c f = false)
    ∧ ∀ y : node,
        y ∈ ys -> ∃ c : node, c ∈ cs ∧ y ∈ field_used_as ctx c f
  end.
Proof.
  unfold get_one_field_usage_of_constructors.
  destruct (existsb (fun c => field_used_top ctx c f) cs) eqn:Htop.
  { apply existsb_exists in Htop as (c & Hc & Htop). exists c. auto. }
  assert (Hnot : ∀ c : node, c ∈ cs -> field_used_top ctx c f = false).
  { intros c Hc. apply not_true_iff_false. intros H.
    apply not_true_iff_false in Htop. apply Htop. apply existsb_exists.
    exists c. auto. }
  destruct (dedup_nodes (flat_map (fun c => field_used_as ctx c f) cs))
    as [ | y0 ys] eqn:Hl; [ exact I | ].
  split; [ exact Hnot | ]. intros y Hy. rewrite <- Hl in Hy.
  apply In_dedup_nodes, in_flat_map in Hy. exact Hy.
Qed.

(* [follow_field_for_set_of_closures] of a set whose closures all hold
   [from] at the value slot [w]: it holds where [from] is. *)
Lemma follow_field_for_set_at (nodes : list (function_slot * node))
    (w : value_slot) (from : node) (m0 : rewrite_metadata) :
  (∀ (g : function_slot) (c : node),
     (g, c) ∈ nodes ->
     ∀ b : node, g_constructor G c (Field_value_slot w) b <-> b = from) ->
  follow_field_for_set_of_closures ctx nodes w = Some m0 ->
  metadata_at sol m0 from.
Proof.
  intros Hheld. pose proof least_closed_tables as Hc.
  unfold follow_field_for_set_of_closures.
  destruct (forallb _ nodes);
    [ intros [= <-]; intros a _ _; exact I | ].
  destruct nodes as [ | [g0 c0] rest];
    [ intros [= <-]; intros a _ _; exact I | ].
  assert (Hcons : ∀ c : node,
            c ∈ map snd ((g0, c0) :: rest) ->
            g_constructor G c (Field_value_slot w) from).
  { intros c Hcin. apply in_map_iff in Hcin as ([g c'] & Hc' & Hin).
    cbn [snd] in Hc'. subst c'. apply (Hheld g c Hin). reflexivity. }
  pose proof (get_single_field_source_spec ctx c0 (Field_value_slot w)) as Hs.
  destruct (get_single_field_source ctx c0 (Field_value_slot w))
    as [ | s | ]; [ discriminate | | ].
  - intros [= <-]. intros a Ha Hha. cbn [describes].
    destruct (Hs from (Hcons c0 (or_introl eq_refl))) as [Hnas Hone].
    destruct Ha as (_ & [Has | Hsrc] & _);
      [ exfalso; exact (bool_both _ Has Hnas) | ].
    exact (Hone a Hsrc Hha).
  - pose proof (get_one_field_usage_of_constructors_spec
                  (map snd ((g0, c0) :: rest)) (Field_value_slot w)) as Hu.
    destruct (get_one_field_usage_of_constructors ctx
                (map snd ((g0, c0) :: rest)) (Field_value_slot w))
      as [ | ys | ].
    + intros [= <-]. intros a Ha _. cbn [describes].
      destruct Ha as (_ & _ & _ & Hau & _). apply Hau.
      destruct Hu as (c & Hcin & Htop).
      exact (field_used_top_any_usage c from _ Htop (Hcons c Hcin)).
    + intros [= <-]. intros a Ha Hha. cbn [describes].
      destruct (s_any_usage sol a) eqn:Hau; [ left; reflexivity | right ].
      intros z Hz. apply get_direct_usages_iff in Hz as (y & Hy & Hyz & Hhz).
      destruct Hu as [Hnot Hys]. destruct (Hys y Hy) as (c & Hcin & Hyc).
      destruct (field_used_as_alias c from y _ Hyc (Hcons c Hcin)
                  (Hnot c Hcin)) as (Hhy & Hnay & Halias).
      destruct Ha as (_ & _ & _ & Hau_n & Hus_n).
      assert (Hnaf : s_any_usage sol from = false).
      { apply not_true_iff_false. intros H. rewrite (Hau_n H) in Hau.
        discriminate Hau. }
      pose proof (usages_alias _ _ _ Hc y from z Hyz Hnay Halias Hnaf)
        as Hfz.
      pose proof (has_usage_alias _ _ _ Hc y from Hhy Halias) as Hhf.
      destruct (Hus_n Hhf z Hfz) as [Haz | Haz];
        [ split; assumption | exfalso; exact (bool_both _ Haz Hau) ].
    + intros [= <-]. intros a _ _. exact I.
Qed.

(* It has an answer where [from] has a source. *)
Lemma follow_field_for_set_some (nodes : list (function_slot * node))
    (w : value_slot) (from : node) :
  (∀ (g : function_slot) (c : node),
     (g, c) ∈ nodes ->
     ∀ b : node, g_constructor G c (Field_value_slot w) b <-> b = from) ->
  s_has_source sol from = true ->
  follow_field_for_set_of_closures ctx nodes w <> None.
Proof.
  intros Hheld Hhs. unfold follow_field_for_set_of_closures.
  destruct (forallb _ nodes); [ discriminate | ].
  destruct nodes as [ | [g0 c0] rest]; [ discriminate | ].
  pose proof (get_single_field_source_spec ctx c0 (Field_value_slot w)) as Hs.
  destruct (get_single_field_source ctx c0 (Field_value_slot w))
    as [ | s | ]; [ | discriminate | ].
  - exfalso.
    destruct (Hs from (proj2 (Hheld g0 c0 (or_introl eq_refl) from) eq_refl))
      as [Hnas Hno].
    destruct (has_source_sources from Hhs) as [Has | (y & Hy & Hhy)];
      [ exact (bool_both _ Has Hnas) | exact (Hno y Hy Hhy) ].
  - destruct (get_one_field_usage_of_constructors ctx _ _); discriminate.
Qed.

End Least_facts.

(* ======================================================================== *)
(* What [follow_field] and [metadata_for_name] keep                         *)
(* ======================================================================== *)

Section Metadata.

Variable ctx : rewrite_context.
Variable is_local : field -> bool.

Local Abbreviation G := (graph_of_facts (rc_facts ctx)).
Local Abbreviation sol := (solution_of_tables (rc_tables ctx)).

Hypothesis Hc : closed is_local G sol.

(* A field that the allocation does not use: [follow_field] forgets. *)
Lemma follow_field_forgets (m : rewrite_metadata) (a from : node)
    (f : field) :
  describes sol m a -> forgets m = false -> g_constructor G a f from ->
  is_local f = false -> s_field_used sol a f = false ->
  forgets (follow_field ctx m f) = true.
Proof.
  intros Hd Hm Hcons Hloc Hfu.
  assert (Hnot_any : s_any_usage sol a = false).
  { apply not_true_iff_false. intros Hau.
    rewrite (field_used_any_usage _ _ _ Hc a f from Hcons Hau Hloc) in Hfu.
    discriminate Hfu. }
  destruct m as [ | source | | usages | ]; cbn [describes forgets] in Hd, Hm;
    try discriminate;
    try contradiction.
  - subst source. simpl. unfold field_used. cbn [solution_of_tables
      s_field_used] in Hfu. rewrite Hfu. reflexivity.
  - rewrite Hnot_any in Hd. discriminate Hd.
  - simpl. unfold for_usages.
    pose proof (get_one_field_usage_spec ctx f usages) as Hs.
    destruct (get_one_field_usage ctx f usages) as [ | ys | ];
      [ | | reflexivity ].
    + destruct Hs as (u & y & Hu & Hy & Hau).
      destruct Hd as [Hau' | Hus]; [ congruence | ].
      destruct (Hus u Hu) as [Hau_u _].
      rewrite (field_used_usage _ _ _ Hc a f from u y Hcons Hau_u Hy
                 (has_usage_any _ _ _ Hc y Hau)) in Hfu.
      discriminate Hfu.
    + destruct Hs as (Hne & _ & Hys). destruct ys as [ | y ys];
        [ contradiction | ].
      destruct (Hys y (or_introl eq_refl)) as (Hhu & u & Hu & Hy).
      destruct Hd as [Hau' | Hus]; [ congruence | ].
      destruct (Hus u Hu) as [Hau_u _].
      rewrite (field_used_usage _ _ _ Hc a f from u y Hcons Hau_u Hy Hhu)
        in Hfu.
      discriminate Hfu.
Qed.

(* The metadata of the field read by the uses given, of an allocation
   that has each as a use. *)
Lemma for_usages_at (usages : list node) (a from : node) (f : field) :
  s_any_usage sol a = false ->
  (∀ u : node,
     u ∈ usages -> s_usages sol a u = true ∧ s_has_usage sol u = true) ->
  g_constructor G a f from ->
  metadata_at sol (for_usages ctx usages f) from.
Proof.
  intros Hau Hus Hcons b Hb Hhsb. unfold for_usages.
  destruct Hb as (_ & _ & _ & Hau_b & Hus_b).
  pose proof (get_one_field_usage_spec ctx f usages) as Hs.
  destruct (get_one_field_usage ctx f usages) as [ | ys | ];
    cbn [describes]; [ | | exact I ].
  - destruct Hs as (u & y & Hu & Hy & Hauy).
    destruct (Hus u Hu) as [Hau_u _].
    pose proof (alias_constructor_usage _ _ _ Hc a u f y from Hcons Hau_u Hau
                  Hy) as Halias.
    exact (Hau_b (any_usage_alias _ _ _ Hc y from Hauy Halias)).
  - destruct (s_any_usage sol b) eqn:Hab; [ left; reflexivity | right ].
    intros z Hz. apply get_direct_usages_iff in Hz as (y & Hy & Hyz & Hhz).
    destruct Hs as (_ & Hnot_any & Hys).
    destruct (Hys y Hy) as (Hhy & u & Hu & Hacc).
    destruct (Hus u Hu) as [Hau_u _].
    pose proof (alias_constructor_usage _ _ _ Hc a u f y from Hcons Hau_u Hau
                  Hacc) as Halias.
    assert (Hau_from : s_any_usage sol from = false).
    { apply not_true_iff_false. intros H. discriminate (Hau_b H). }
    pose proof (usages_alias _ _ _ Hc y from z Hyz (Hnot_any u y Hu Hacc)
                  Halias Hau_from) as Hfz.
    pose proof (has_usage_alias _ _ _ Hc y from Hhy Halias) as Hhf.
    destruct (Hus_b Hhf z Hfz) as [Hbz | Hbz];
      [ split; assumption | discriminate Hbz ].
Qed.

(* A field that the allocation uses: [follow_field] holds where the field
   is held. *)
Lemma follow_field_at (m : rewrite_metadata) (a from : node) (f : field) :
  describes sol m a -> forgets m = false -> g_constructor G a f from ->
  is_local f = false -> s_field_used sol a f = true ->
  metadata_at sol (follow_field ctx m f) from.
Proof.
  intros Hd Hm Hcons Hloc Hfu.
  assert (Hany : s_any_usage sol a = true ->
                 metadata_at sol (follow_field ctx m f) from ->
                 metadata_at sol (follow_field ctx m f) from) by auto.
  destruct m as [ | source | | usages | ]; cbn [describes forgets] in Hd, Hm;
    try discriminate;
    try contradiction.
  - subst source. simpl. unfold field_used.
    cbn [solution_of_tables s_field_used] in Hfu. rewrite Hfu. simpl.
    pose proof (get_single_field_source_spec ctx a f) as Hs.
    destruct (get_single_field_source ctx a f) as [ | s | ].
    + intros b Hb Hhsb. exfalso.
      destruct (Hs from Hcons) as [Hnas Hno].
      destruct Hb as (_ & [Has | Hsrc] & _); [ congruence | ].
      exact (Hno b Hsrc Hhsb).
    + intros b Hb Hhsb. cbn [describes].
      destruct (Hs from Hcons) as [Hnas Hone].
      destruct Hb as (_ & [Has | Hsrc] & _); [ congruence | ].
      exact (Hone b Hsrc Hhsb).
    + pose proof (get_usages_spec ctx a) as Hu.
      destruct (get_usages ctx a) as [ | usages | ].
      * intros b Hb _. destruct Hb as (_ & _ & _ & Hau_b & _).
        exact (Hau_b (any_usage_constructor _ _ _ Hc a f from (proj1 Hu)
                        Hcons Hloc)).
      * destruct Hu as (_ & Hau & Hiff).
        apply (for_usages_at usages a from f Hau); [ | exact Hcons ].
        intros u Hu. apply Hiff. exact Hu.
      * intros b _ _. exact I.
  - intros b Hb _. destruct Hb as (_ & _ & _ & Hau_b & _).
    exact (Hau_b (any_usage_constructor _ _ _ Hc a f from Hd Hcons Hloc)).
  - destruct (s_any_usage sol a) eqn:Hau.
    + intros b Hb _. destruct Hb as (_ & _ & _ & Hau_b & _).
      pose proof (Hau_b (any_usage_constructor _ _ _ Hc a f from Hau Hcons
                           Hloc)) as Hab.
      cbn [follow_field]. unfold for_usages.
      destruct (get_one_field_usage ctx f usages); cbn [describes];
        [ exact Hab | left; exact Hab | exact I ].
    + destruct Hd as [Hd | Hd]; [ discriminate Hd | ].
      exact (for_usages_at usages a from f Hau Hd Hcons).
Qed.

(* The metadata of a name holds at its node, which has a use unless the
   metadata forgets. *)
Lemma metadata_for_name_at (x : node) :
  metadata_at sol (metadata_for_name ctx x) x
  ∧ (forgets (metadata_for_name ctx x) = false -> s_has_usage sol x = true).
Proof.
  unfold metadata_for_name. pose proof (get_single_source_spec ctx x) as Hs.
  destruct (get_single_source ctx x) as [ | y | ].
  - pose proof (get_usages_spec ctx x) as Hu.
    destruct (get_usages ctx x) as [ | us | ].
    + split; [ | intros _; exact (proj2 Hu) ].
      intros a Ha _. destruct Ha as (_ & _ & _ & Hau & _).
      exact (Hau (proj1 Hu)).
    + destruct Hu as (Hhu & _ & Hiff).
      split; [ | intros _; exact Hhu ].
      intros a Ha _. destruct Ha as (_ & _ & _ & _ & Hus).
      cbn [describes].
      destruct (s_any_usage sol a) eqn:Hau; [ left; reflexivity | ].
      right. intros u Hu. apply Hiff in Hu as [Hxu Hhuu].
      destruct (Hus Hhu u Hxu) as [H | H]; [ auto | congruence ].
    + split; [ intros a _ _; exact I | discriminate ].
  - destruct (has_use ctx x) eqn:Hhu.
    + split; [ | intros _; exact Hhu ].
      intros a Ha Hhsa. cbn [describes]. destruct Hs as (_ & Hnas & Hone).
      destruct Ha as (_ & [Has | Hsrc] & _); [ congruence | ].
      exact (Hone a Hsrc Hhsa).
    + split; [ intros a _ _; exact I | discriminate ].
  - split; [ | discriminate ].
    intros a Ha _. destruct Ha as (Hhs & _). congruence.
Qed.

End Metadata.

(* ======================================================================== *)
(* The join of metadata                                                     *)
(* ======================================================================== *)

Section Join.

Variable ctx : rewrite_context.

Local Abbreviation sol := (solution_of_tables (rc_tables ctx)).

(* What [usages_of] in [join_metadata] keeps of a metadata. *)
Lemma usages_of_metadata_describes (m : rewrite_metadata) (a : node) :
  describes sol m a ->
  match usages_of_metadata ctx m with
  | None => s_any_usage sol a = true
  | Some us =>
    s_any_usage sol a = true
    ∨ ∀ u : node,
        u ∈ us -> s_usages sol a u = true ∧ s_has_usage sol u = true
  end.
Proof.
  destruct m as [ | s | | us | ]; cbn [describes usages_of_metadata].
  - intros [].
  - intros <-. pose proof (get_usages_spec ctx a) as Hs.
    destruct (get_usages ctx a) as [ | us | ].
    + exact (proj1 Hs).
    + right. intros u Hu. apply (proj2 (proj2 Hs)). exact Hu.
    + right. intros u [].
  - intros H. exact H.
  - intros H. exact H.
  - intros _. right. intros u [].
Qed.

Lemma join_usages_describes (x y : rewrite_metadata) (a : node) :
  describes sol x a ∨ describes sol y a ->
  describes sol (join_usages ctx x y) a.
Proof.
  intros H. unfold join_usages.
  pose proof (usages_of_metadata_describes x a) as Hx.
  pose proof (usages_of_metadata_describes y a) as Hy.
  destruct (usages_of_metadata ctx x) as [ux | ],
    (usages_of_metadata ctx y) as [uy | ]; cbn [describes];
    destruct H as [H | H];
    try (specialize (Hx H)); try (specialize (Hy H)).
  - destruct Hx as [Hx | Hx]; [ left; exact Hx | right ].
    intros u Hu. apply Hx. unfold inter_nodes in Hu.
    apply filter_In in Hu. exact (proj1 Hu).
  - destruct Hy as [Hy | Hy]; [ left; exact Hy | right ].
    intros u Hu. apply Hy. unfold inter_nodes in Hu.
    apply filter_In in Hu as [_ Hu]. apply mem_node_In. exact Hu.
  - exact Hx.
  - left. exact Hy.
  - left. exact Hx.
  - exact Hy.
  - exact Hx.
  - exact Hy.
Qed.

(* The join of two metadata describes what either describes. *)
Lemma join_metadata_describes (x y : rewrite_metadata) (a : node) :
  describes sol x a ∨ describes sol y a ->
  describes sol (join_metadata ctx x y) a.
Proof.
  intros H.
  destruct x as [ | s1 | | u1 | ], y as [ | s2 | | u2 | ];
    cbn [join_metadata];
    try (apply join_usages_describes; exact H);
    cbn [describes] in H |- *; try tauto.
  destruct (node_eqb s1 s2) eqn:Heq.
  - apply node_eqb_eq in Heq. subst s2. cbn [describes].
    destruct H as [H | H]; exact H.
  - apply join_usages_describes. exact H.
Qed.

(* The metadata of the slots of the join of sets of closures describe
   what those of one of the sets describe. *)
Definition slots_covered (acc e : slots_metadata) : Prop :=
  (∀ (w : value_slot) (mw : rewrite_metadata),
     lookup w (fst acc) = Some mw ->
     ∃ me : rewrite_metadata,
       lookup w (fst e) = Some me
       ∧ ∀ a : node, describes sol me a -> describes sol mw a)
  ∧ ∀ (g : function_slot) (p : rewrite_metadata * bool),
      lookup g (snd acc) = Some p -> is_bound g (snd e) = true.

Lemma covered_refl (e : slots_metadata) : slots_covered e e.
Proof.
  split.
  - intros w mw Hw. exists mw. auto.
  - intros g p Hg. apply is_bound_lookup. exists p. exact Hg.
Qed.

Lemma covered_join_left (acc b e : slots_metadata) :
  slots_covered acc e -> slots_covered (join_slots ctx acc b) e.
Proof.
  intros [Hv Hf]. split.
  - intros w mw Hw. cbn [join_slots fst] in Hw.
    apply lookup_inter_slots in Hw as (ma & mb & Ha & Hb & ->).
    destruct (Hv w ma Ha) as (me & Hme & Hd). exists me.
    split; [ exact Hme | ].
    intros a H. apply join_metadata_describes. left. exact (Hd a H).
  - intros g p Hg. cbn [join_slots snd] in Hg.
    apply lookup_inter_slots in Hg as (pa & pb & Ha & _ & _).
    exact (Hf g pa Ha).
Qed.

Lemma covered_join_right (acc b : slots_metadata) :
  slots_covered (join_slots ctx acc b) b.
Proof.
  split.
  - intros w mw Hw. cbn [join_slots fst] in Hw.
    apply lookup_inter_slots in Hw as (ma & mb & _ & Hb & ->).
    exists mb. split; [ exact Hb | ].
    intros a H. apply join_metadata_describes. right. exact H.
  - intros g p Hg. cbn [join_slots snd] in Hg.
    apply lookup_inter_slots in Hg as (pa & pb & _ & Hb & _).
    apply is_bound_lookup. exists pb. exact Hb.
Qed.

Lemma fold_join_keeps (rest : list slots_metadata) (acc e : slots_metadata) :
  slots_covered acc e ->
  slots_covered (List.fold_left (join_slots ctx) rest acc) e.
Proof.
  revert acc. induction rest as [ | b rest IH]; intros acc H; [ exact H | ].
  cbn [List.fold_left]. apply IH. apply covered_join_left. exact H.
Qed.

Lemma fold_join_covered (rest : list slots_metadata)
    (first e : slots_metadata) :
  e ∈ first :: rest ->
  slots_covered (List.fold_left (join_slots ctx) rest first) e.
Proof.
  revert first. induction rest as [ | b rest IH]; intros first He.
  - destruct He as [<- | []]. apply covered_refl.
  - cbn [List.fold_left]. destruct He as [<- | [<- | He]].
    + apply fold_join_keeps. apply covered_join_left. apply covered_refl.
    + apply fold_join_keeps. apply covered_join_right.
    + apply IH. right. exact He.
Qed.

End Join.

(* ======================================================================== *)
(* Values rebuilt                                                           *)
(* ======================================================================== *)

Section Rebuilt_values.

Variable G : graph.
Variable sol : solution.

Lemma rebuilt_value_kind (v : value) (n : node) (v' : value) :
  rebuilt_value G sol v n v' -> kind_of_value v' = kind_of_value v.
Proof.
  destruct v; cbn [rebuilt_value]; try contradiction.
  - intros [_ ->]. reflexivity.
  - intros (sites & vs & -> & _). reflexivity.
  - intros [-> _]. reflexivity.
  - intros (a & fields' & _ & _ & -> & _). reflexivity.
  - intros (elements' & ->). reflexivity.
  - intros (a & bn & from & _ & _ & _ & _ & ->).
    destruct (s_field_used sol a (Field_boxed_number bn)); reflexivity.
  - intros (closure' & args' & ->). reflexivity.
Qed.

Lemma rebuilt_under_kind (m : rewrite_metadata) (v v' : value) :
  rebuilt_under G sol m v v' -> kind_of_value v' = kind_of_value v.
Proof.
  intros [(c & -> & ->) | (n & Hv & _)];
    [ reflexivity | exact (rebuilt_value_kind v n v' Hv) ].
Qed.

Lemma rebuilt_under_const (m : rewrite_metadata) (c : const) (v' : value) :
  rebuilt_under G sol m (Value_const c) v' -> v' = Value_const c.
Proof.
  intros [(c' & [= <-] & ->) | (n & [_ ->] & _)]; reflexivity.
Qed.

Lemma rebuilt_under_unallocated (m : rewrite_metadata) (v v' : value) :
  unallocated v -> rebuilt_under G sol m v v' -> v' = v.
Proof.
  intros [(c & ->) | (s & ->)] Hv; [ exact (rebuilt_under_const m c v' Hv) | ].
  destruct Hv as [(c & [=] & _) | (n & [-> _] & _)]. reflexivity.
Qed.

Lemma rebuilt_under_not_null (m : rewrite_metadata) (v v' : value) :
  rebuilt_under G sol m v v' -> v <> Value_const Null ->
  v' <> Value_const Null.
Proof.
  intros [(c & -> & ->) | (n & Hv & _)] Hnull; [ exact Hnull | ].
  destruct v; cbn [rebuilt_value] in Hv; try contradiction.
  - destruct Hv as [_ ->]. exact Hnull.
  - destruct Hv as (sites & vs & -> & _). discriminate.
  - destruct Hv as [-> _]. discriminate.
  - destruct Hv as (a & fields' & _ & _ & -> & _). discriminate.
  - destruct Hv as (elements' & ->). discriminate.
  - destruct Hv as (a & bn & from & _ & _ & _ & _ & ->).
    destruct (s_field_used sol a (Field_boxed_number bn)); discriminate.
  - destruct Hv as (closure' & args' & ->). discriminate.
Qed.

Lemma fields_rebuilt_length (P : nat -> value -> value -> Prop) (k : nat)
    (fields fields' : list value) :
  fields_rebuilt P k fields fields' -> length fields' = length fields.
Proof.
  revert k fields'. induction fields as [ | field fields IH];
    intros k [ | field' fields'] H; cbn [fields_rebuilt] in H;
    try contradiction; [ reflexivity | ].
  cbn [length]. rewrite (IH (S k) fields' (proj2 H)). reflexivity.
Qed.

Lemma field_rebuilt_kind (a : node) (i : nat) (field field' : value) :
  field_rebuilt G sol (rebuilt_value G sol) a i field field' ->
  kind_of_value field' = kind_of_value field.
Proof.
  intros (from & _ & Hused & Hunused).
  destruct (s_field_used sol a (Field_block i (kind_of_value field)))
    eqn:Hfu.
  - exact (rebuilt_value_kind field from field' (proj2 (Hused eq_refl))).
  - rewrite (Hunused eq_refl). reflexivity.
Qed.

Lemma fields_rebuilt_kinds (a : node) (k : nat) (fields fields' : list value) :
  fields_rebuilt (field_rebuilt G sol (rebuilt_value G sol) a) k fields
    fields' ->
  map kind_of_value fields' = map kind_of_value fields.
Proof.
  revert k fields'. induction fields as [ | field fields IH];
    intros k [ | field' fields'] H; cbn [fields_rebuilt] in H;
    try contradiction; [ reflexivity | ].
  destruct H as [Hfield Hfields]. cbn [map].
  rewrite (field_rebuilt_kind a k field field' Hfield),
    (IH (S k) fields' Hfields).
  reflexivity.
Qed.

End Rebuilt_values.

(* ======================================================================== *)
(* The fragment                                                             *)
(* ======================================================================== *)

Lemma eval_const_unallocated (vars : env) (c : const) (w : value) :
  eval_simple vars (Const c) = Some w -> unallocated w.
Proof. simpl. intros [= <-]. left. exists c. reflexivity. Qed.

Lemma unknown_fragment (vars : env) (k : flambda_kind) :
  type_in_fragment vars (unknown k).
Proof.
  split; [ destruct k as [ | [] | | ]; reflexivity | ].
  intros s k' w Hin. destruct k as [ | [] | | ]; destruct Hin.
Qed.

Lemma bottom_fragment (vars : env) (k : flambda_kind) :
  type_in_fragment vars (bottom k).
Proof.
  split; [ destruct k as [ | [] | | ]; reflexivity | ].
  intros s k' w Hin. destruct k as [ | [] | | ]; destruct Hin.
Qed.

(* An alias of a type is among its aliases. *)
Lemma get_alias_in_aliases (t : flambda_type) (s : simple) :
  get_alias_opt t = Some s -> (s, kind t) ∈ aliases_with_kinds t.
Proof.
  destruct t as [d | d | k d | d | d]; destruct d as [ | [h | s'] | ];
    simpl; try discriminate; intros [= <-]; left; reflexivity.
Qed.

(* The head of a type of kind value that is in the fragment: what it holds
   is. *)
Definition non_null_in_fragment (vars : env)
    (head : head_of_kind_value_non_null) : Prop :=
  shaped_non_null head = true
  ∧ ∀ (s : simple) (k : flambda_kind) (w : value),
      (s, k) ∈ aliases_of_head_of_kind_value_non_null head ->
      eval_simple vars s = Some w -> unallocated w.

Lemma head_fragment (vars : env) (head : head_of_kind_value_non_null)
    (is_null : is_null) :
  type_in_fragment vars
    (Type_value (Ok (No_alias (Head_of_kind_value (Ok head) is_null)))) ->
  non_null_in_fragment vars head.
Proof.
  intros [Hcf Hal]. split; [ exact Hcf | ].
  intros s k w Hin. apply (Hal s k w). cbn [aliases_with_kinds
    aliases_of_descr aliases_of_head_of_kind_value].
  apply in_or_app. left. exact Hin.
Qed.

Lemma immediates_fragment (vars : env) (is_int : option variable)
    (ty : flambda_type) (get_tag : option variable)
    (blocks : Or_unknown.t row_like_for_blocks)
    (extensions : variant_extensions) (is_unique : bool) :
  non_null_in_fragment vars
    (Variant is_int (Or_unknown.Known ty) get_tag blocks extensions
       is_unique) ->
  type_in_fragment vars ty.
Proof.
  intros [Hcf Hal]. cbn [shaped_non_null] in Hcf.
  apply andb_true_iff in Hcf as [Hcf _]. split; [ exact Hcf | ].
  intros s k w Hin. apply (Hal s k w).
  cbn [aliases_of_head_of_kind_value_non_null].
  apply in_or_app. right. apply in_or_app. left. exact Hin.
Qed.

Lemma blocks_fragment (vars : env) (is_int : option variable)
    (immediates : Or_unknown.t flambda_type) (get_tag : option variable)
    (blocks : row_like_for_blocks) (extensions : variant_extensions)
    (is_unique : bool) :
  non_null_in_fragment vars
    (Variant is_int immediates get_tag (Or_unknown.Known blocks) extensions
       is_unique) ->
  shaped_blocks blocks = true
  ∧ ∀ (s : simple) (k : flambda_kind) (w : value),
      (s, k) ∈ aliases_of_row_like_for_blocks blocks ->
      eval_simple vars s = Some w -> unallocated w.
Proof.
  intros [Hcf Hal]. cbn [shaped_non_null] in Hcf.
  apply andb_true_iff in Hcf as [_ Hcf]. split; [ exact Hcf | ].
  intros s k w Hin. apply (Hal s k w).
  cbn [aliases_of_head_of_kind_value_non_null].
  apply in_or_app. right. apply in_or_app. right.
  apply in_or_app. right. apply in_or_app. left. exact Hin.
Qed.

(* The fields of a case of blocks in the fragment are. *)
Lemma block_case_fragment (vars : env) (case : row_like_block_case) :
  forallb shaped_type (case_maps_to case) = true ->
  (∀ (s : simple) (k : flambda_kind) (w : value),
     (s, k) ∈ flat_map aliases_with_kinds (case_maps_to case) ->
     eval_simple vars s = Some w -> unallocated w) ->
  ∀ ty : flambda_type, ty ∈ case_maps_to case -> type_in_fragment vars ty.
Proof.
  intros Hcf Hal ty Hty. split.
  - rewrite forallb_forall in Hcf. exact (Hcf ty Hty).
  - intros s k w Hin. apply (Hal s k w). apply in_flat_map.
    exists ty. auto.
Qed.

Lemma known_case_fragment (vars : env)
    (known_tags : Keyed.finite_map Z (Or_unknown.t row_like_block_case))
    (other_tags : Or_bottom.t row_like_block_case) (alloc_mode : Alloc_mode.t)
    (tag : Z) (case : row_like_block_case) :
  shaped_blocks (Row_like_for_blocks known_tags other_tags alloc_mode)
  = true ->
  (∀ (s : simple) (k : flambda_kind) (w : value),
     (s, k)
     ∈ aliases_of_row_like_for_blocks
         (Row_like_for_blocks known_tags other_tags alloc_mode) ->
     eval_simple vars s = Some w -> unallocated w) ->
  (tag, Or_unknown.Known case) ∈ known_tags ->
  ∀ ty : flambda_type, ty ∈ case_maps_to case -> type_in_fragment vars ty.
Proof.
  intros Hcf Hal Hin. cbn [shaped_blocks] in Hcf.
  apply andb_true_iff in Hcf as [Hcf _]. rewrite forallb_forall in Hcf.
  apply block_case_fragment.
  - exact (Hcf _ Hin).
  - intros s k w Hs. apply (Hal s k w).
    cbn [aliases_of_row_like_for_blocks]. apply in_or_app. left.
    apply in_flat_map. exists (tag, Or_unknown.Known case).
    split; [ exact Hin | ]. cbn [snd aliases_of_block_case].
    apply in_or_app. left. exact Hs.
Qed.

Lemma other_case_fragment (vars : env)
    (known_tags : Keyed.finite_map Z (Or_unknown.t row_like_block_case))
    (case : row_like_block_case) (alloc_mode : Alloc_mode.t) :
  shaped_blocks
    (Row_like_for_blocks known_tags (Or_bottom.Ok case) alloc_mode)
  = true ->
  (∀ (s : simple) (k : flambda_kind) (w : value),
     (s, k)
     ∈ aliases_of_row_like_for_blocks
         (Row_like_for_blocks known_tags (Or_bottom.Ok case) alloc_mode) ->
     eval_simple vars s = Some w -> unallocated w) ->
  ∀ ty : flambda_type, ty ∈ case_maps_to case -> type_in_fragment vars ty.
Proof.
  intros Hcf Hal. cbn [shaped_blocks] in Hcf.
  apply andb_true_iff in Hcf as [_ Hcf].
  apply block_case_fragment; [ exact Hcf | ].
  intros s k w Hs. apply (Hal s k w).
  cbn [aliases_of_row_like_for_blocks]. apply in_or_app. right.
  cbn [aliases_of_block_case]. apply in_or_app. left. exact Hs.
Qed.

Lemma contents_fragment (vars : env) (k : boxable_number)
    (contents : flambda_type) (alloc_mode : Alloc_mode.t) :
  non_null_in_fragment vars (Boxed_number k contents alloc_mode) ->
  type_in_fragment vars contents.
Proof.
  intros [Hcf Hal]. split; [ exact Hcf | ].
  intros s k' w Hin. exact (Hal s k' w Hin).
Qed.

(* A type that is not an alias, expanded, is itself or unknown or
   bottom. *)
Lemma of_non_alias_fragment (vars : env) (ty : flambda_type)
    (e : Expanded_type.t) :
  Expanded_type.of_non_alias_type ty = Some e ->
  type_in_fragment vars ty -> type_in_fragment vars (Expanded_type.to_type e).
Proof.
  destruct ty as [d | d | k d | d | d]; destruct d as [ | [h | s] | ];
    cbn; intros He; try discriminate; injection He as <-; cbn;
    intros Hfrag; try destruct k;
    first [ exact Hfrag
          | split; [ reflexivity | intros s' k' w Hin; cbn in Hin;
                                   destruct Hin ] ].
Qed.

(* The expanded type of a constant. *)
Lemma create_const_fragment (vars : env) (c : const) :
  type_in_fragment vars (Expanded_type.to_type (Expanded_type.create_const c)).
Proof.
  destruct c as [ | | | | | | | | | | | | | | pk ];
    [ .. | destruct pk as [ | [] | | ] ]; cbn; split;
    try reflexivity; intros s k w Hin Hev; cbn in Hin;
    repeat (destruct Hin as [Hin | Hin];
            [ injection Hin as <- _; exact (eval_const_unallocated _ _ _ Hev)
            | ]);
    destruct Hin.
Qed.

Section Env_fragment.

Variable vars : env.
Variable tenv : Typing_env.t.

Hypothesis Hfrag : env_in_fragment vars tenv.

Lemma any_non_null_value_fragment : type_in_fragment vars any_non_null_value.
Proof.
  split; [ reflexivity | ]. intros s k w Hin. cbn in Hin. destruct Hin.
Qed.

(* What the environment has for a name is of the fragment. *)
Lemma find_fragment (n : name) (k : option flambda_kind) (ty : flambda_type) :
  Typing_env.find tenv n k = Some ty -> type_in_fragment vars ty.
Proof.
  unfold Typing_env.find, Typing_env.find_with_binding_time_and_mode'.
  destruct (Keyed.lookup name_eqb n (Typing_env.names_to_types tenv))
    as [[ty0 bt] | ] eqn:Hl.
  - destruct (Typing_env.check_optional_kind_matches ty0 k); [ | discriminate ].
    destruct (Typing_env.is_bottom tenv); cbn; intros [= <-].
    + apply bottom_fragment.
    + exact (Hfrag n ty0 bt Hl).
  - destruct n as [x | s]; [ discriminate | ].
    destruct (among s (Typing_env.defined_symbols tenv)
              && Typing_env.check_optional_kind_matches
                   (fst Typing_env.initial_symbol_type) k);
      [ | discriminate ].
    cbn. intros [= <-]. apply any_non_null_value_fragment.
Qed.

(* The head of a type of the fragment is of the fragment. *)
Lemma expand_head_fragment (t : flambda_type) (e : Expanded_type.t) :
  expand_head tenv t = Some e -> type_in_fragment vars t ->
  type_in_fragment vars (Expanded_type.to_type e).
Proof.
  unfold expand_head. destruct (get_alias_opt t) as [s | ] eqn:Ha.
  - destruct (Typing_env.get_canonical_simple_exn tenv
                (Some Name_mode.in_types) None s)
      as [[c | ] | ]; cbn; [ | | discriminate ].
    + unfold expand_head_of_alias_type. destruct c as [x | sym | c].
      * destruct (Typing_env.find tenv (Name_var x) (Some (kind t)))
          as [ty | ] eqn:Hf; cbn; [ | discriminate ].
        intros He _. exact (of_non_alias_fragment vars ty e He
                              (find_fragment _ _ ty Hf)).
      * destruct (Typing_env.find tenv (Name_symbol sym) (Some (kind t)))
          as [ty | ] eqn:Hf; cbn; [ | discriminate ].
        intros He _. exact (of_non_alias_fragment vars ty e He
                              (find_fragment _ _ ty Hf)).
      * intros [= <-] _. apply create_const_fragment.
    + intros He _. apply (of_non_alias_fragment vars _ e He).
      apply unknown_fragment.
  - intros He Ht. exact (of_non_alias_fragment vars t e He Ht).
Qed.

End Env_fragment.

Lemma type_for_const_fragment (vars : env) (c : const) :
  type_in_fragment vars (type_for_const c).
Proof.
  destruct c as [ | | | | | | | | | | | | | | pk ];
    [ .. | destruct pk as [ | [] | | ] ]; cbn; split;
    try reflexivity; intros s k w Hin Hev; cbn in Hin;
    repeat (destruct Hin as [Hin | Hin];
            [ injection Hin as <- _; exact (eval_const_unallocated _ _ _ Hev)
            | ]);
    destruct Hin.
Qed.

(* An alias is of the shape of the fragment. *)
Lemma alias_shaped (t : flambda_type) :
  is_alias t = true -> shaped_type t = true.
Proof.
  unfold is_alias.
  destruct t as [d | d | k d | d | d]; destruct d as [ | [h | s] | ];
    cbn; try discriminate; reflexivity.
Qed.

(* A case of closures of a type of the fragment: its slots are among those
   of its index, its closure types are aliases, and all its types are of
   the fragment. *)
Lemma closures_case_fragment (vars : env)
    (known : finite_map row_like_closures_case) (alloc_mode : Alloc_mode.t)
    (is_null : is_null) (slot : function_slot) (case : row_like_closures_case)
    (fts : finite_map (Or_unknown.t function_type))
    (cts vsts : finite_map flambda_type) :
  type_in_fragment vars
    (Type_value
       (Ok
          (No_alias
             (Head_of_kind_value
                (Ok (Closures (Row_like_for_closures known) alloc_mode))
                is_null)))) ->
  (slot, case) ∈ known -> case_maps_to case = Closures_entry fts cts vsts ->
  FlambdaMapSpec.all_among (domain cts)
    (contents_closures (closures_index_contents (case_index case)))
  = true
  ∧ FlambdaMapSpec.all_among (domain vsts)
      (contents_value_slots (closures_index_contents (case_index case)))
    = true
  ∧ (∀ (g : function_slot) (ty : flambda_type),
       (g, ty) ∈ cts -> is_alias ty = true ∧ type_in_fragment vars ty)
  ∧ (∀ (w : value_slot) (ty : flambda_type),
       (w, ty) ∈ vsts -> type_in_fragment vars ty).
Proof.
  intros [Hsh Hal] Hin Hentry.
  cbn [shaped_type shaped_non_null shaped_closures] in Hsh.
  rewrite forallb_forall in Hsh. specialize (Hsh _ Hin). cbn [snd] in Hsh.
  rewrite Hentry in Hsh.
  apply andb_true_iff in Hsh as [Hsh Hvs].
  apply andb_true_iff in Hsh as [Hsh Hcts].
  apply andb_true_iff in Hsh as [Hcl Hvl].
  assert (Hal_entry : ∀ (s : simple) (k : flambda_kind) (w : value),
             (s, k) ∈ aliases_of_closures_entry (Closures_entry fts cts vsts)
             -> eval_simple vars s = Some w -> unallocated w).
  { intros s k w Hs. apply (Hal s k w).
    cbn [aliases_with_kinds aliases_of_descr aliases_of_head_of_kind_value
      aliases_of_head_of_kind_value_non_null
      aliases_of_row_like_for_closures].
    apply in_or_app. left. apply in_flat_map. exists (slot, case).
    split; [ exact Hin | ]. cbn [snd]. rewrite Hentry.
    apply in_or_app. left. exact Hs. }
  split; [ exact Hcl | split; [ exact Hvl | split ] ].
  - intros g ty Hg. rewrite forallb_forall in Hcts.
    pose proof (Hcts (g, ty) Hg) as Hty. cbn [snd] in Hty.
    split; [ exact Hty | split; [ exact (alias_shaped ty Hty) | ] ].
    intros s k w Hs. apply (Hal_entry s k w). cbn [aliases_of_closures_entry].
    apply in_or_app. left. apply in_flat_map. exists (g, ty). auto.
  - intros w ty Hw. rewrite forallb_forall in Hvs.
    split; [ exact (Hvs (w, ty) Hw) | ].
    intros s k x Hs. apply (Hal_entry s k x). cbn [aliases_of_closures_entry].
    apply in_or_app. right. apply in_flat_map. exists (w, ty). auto.
Qed.


(* The contents of a boxed number of the kind of a field of a boxed
   number. *)
Lemma boxable_kind_eq (bn k : boxable_number) (c0 c : const) :
  is_number_of_kind bn c0 = true -> boxable_contents c0 = Some c ->
  kind_of_const c = unboxed_kind k -> bn = k.
Proof.
  unfold boxable_contents. destruct (boxable_number_of_const c0);
    [ | discriminate ].
  intros Hn [= <-] Hk.
  destruct bn, c0; try discriminate Hn; destruct k; try discriminate Hk;
    reflexivity.
Qed.


(* ======================================================================== *)
(* Closure types                                                            *)
(* ======================================================================== *)

(* What [meet_single_closures_entry] answers of a type: its head, a
   structure of closures with a single case, of the slot and the entry
   answered. *)
Lemma meet_closures_head (tenv : Typing_env.t) (t : flambda_type)
    (slot : function_slot) (am : Alloc_mode.t) (entry : closures_entry)
    (ft : function_type) :
  Provers.meet_single_closures_entry tenv t
  = Some (Meet_shortcut.Known_result (slot, am, entry, ft)) ->
  ∃ (e : Expanded_type.t) (known : finite_map row_like_closures_case)
    (isn : is_null),
    expand_head tenv t = Some e
    ∧ Expanded_type.descr e
      = Ok (Expanded_type.Head_value
              (Head_of_kind_value
                 (Ok (Closures (Row_like_for_closures known) am)) isn))
    ∧ single_closures_entry
        (row_like_for_closures_get_single_tag (Row_like_for_closures known))
      = Some (slot, entry).
Proof.
  unfold Provers.meet_single_closures_entry, Provers.gen_value_to_meet,
    Provers.expand_head.
  destruct (expand_head tenv t) as [e | ] eqn:He; cbn; [ | discriminate ].
  unfold descr_oub_of_expanded.
  destruct (Expanded_type.descr e)
    as [ | [head | head | k' head | head | head] | ] eqn:Hd;
    try (destruct (Expanded_type.kind e) as [ | [] | | ]; discriminate);
    try discriminate.
  destruct head as [[ | h | ] isn]; try discriminate.
  destruct h as [ | | | [known] am' | ]; try (cbn; discriminate).
  unfold Provers.prove_single_closures_entry_generic_value. cbv beta iota.
  match goal with
  | |- context [row_like_for_closures_get_single_tag ?x] =>
    destruct (row_like_for_closures_get_single_tag x)
      as [[ | s en | s en] | ] eqn:Hst
  end; cbn; try discriminate;
    unfold Provers.single_closures_entry_of_singleton;
    destruct (closures_entry_find_function_type en _ s); cbn;
    try discriminate;
    intros [= <- <- <- <-];
    exists e, known, isn;
    (split; [ reflexivity | split; [ exact Hd | ] ]);
    change (single_closures_entry
              (row_like_for_closures_get_single_tag
                 (Row_like_for_closures known))
            = Some (s, en));
    rewrite Hst; reflexivity.
Qed.

(* The single case of a structure of closures. *)
Lemma single_tag_case (known : finite_map row_like_closures_case)
    (slot : function_slot) (entry : closures_entry) :
  single_closures_entry
    (row_like_for_closures_get_single_tag (Row_like_for_closures known))
  = Some (slot, entry) ->
  ∃ case : row_like_closures_case,
    (∀ g : function_slot,
       lookup g known = if Pos.eqb g slot then Some case else None)
    ∧ case_maps_to case = entry.
Proof.
  unfold row_like_for_closures_get_single_tag.
  match goal with
  | |- context [@Keyed.get_singleton_binding ?K ?eqb ?A ?k] =>
    destruct (@Keyed.get_singleton_binding K eqb A k)
      as [[s case] | ] eqn:Hs
  end; cbn; [ | intros H; discriminate H ].
  intros H.
  assert (Hsc : s = slot ∧ case_maps_to case = entry).
  { destruct (index_domain (case_index case)) as [contents | contents];
      destruct (among s (contents_closures contents)); cbn in H;
      try discriminate H; injection H as <- <-; auto. }
  destruct Hsc as [<- Hent]. exists case. split; [ | exact Hent ].
  exact (proj1 (keyed_get_singleton_binding positive Pos.eqb
                  pos_eqb_decides_equality _ s case known) Hs).
Qed.

(* The types that the traversal reads of a value slot and of a function
   slot of a closure type. *)
Lemma destructure_env_var (e : Expanded_type.t)
    (known : finite_map row_like_closures_case) (am : Alloc_mode.t)
    (isn : is_null) (slot : function_slot)
    (fts : finite_map (Or_unknown.t function_type))
    (cts vsts : finite_map flambda_type) (w : value_slot) :
  Expanded_type.descr e
  = Ok (Expanded_type.Head_value
          (Head_of_kind_value
             (Ok (Closures (Row_like_for_closures known) am)) isn)) ->
  single_closures_entry
    (row_like_for_closures_get_single_tag (Row_like_for_closures known))
  = Some (slot, Closures_entry fts cts vsts) ->
  destructure_closure (fun c => row_like_for_closures_get_env_var c w)
    (Some e)
  = match lookup w vsts with
    | Some ty => ty
    | None => unknown Value
    end.
Proof.
  intros He Hs. unfold destructure_closure. rewrite He.
  unfold row_like_for_closures_get_env_var.
  destruct (row_like_for_closures_get_single_tag (Row_like_for_closures known))
    as [[ | s en | s en] | ]; cbn in Hs; try discriminate;
    injection Hs as _ Hen; subst en; destruct (lookup w vsts); reflexivity.
Qed.

Lemma destructure_closure_type (e : Expanded_type.t)
    (known : finite_map row_like_closures_case) (am : Alloc_mode.t)
    (isn : is_null) (slot : function_slot)
    (fts : finite_map (Or_unknown.t function_type))
    (cts vsts : finite_map flambda_type) (g : function_slot) :
  Expanded_type.descr e
  = Ok (Expanded_type.Head_value
          (Head_of_kind_value
             (Ok (Closures (Row_like_for_closures known) am)) isn)) ->
  single_closures_entry
    (row_like_for_closures_get_single_tag (Row_like_for_closures known))
  = Some (slot, Closures_entry fts cts vsts) ->
  destructure_closure (fun c => row_like_for_closures_get_closure c g)
    (Some e)
  = match lookup g cts with
    | Some ty => ty
    | None => unknown Value
    end.
Proof.
  intros He Hs. unfold destructure_closure. rewrite He.
  unfold row_like_for_closures_get_closure.
  destruct (row_like_for_closures_get_single_tag (Row_like_for_closures known))
    as [[ | s en | s en] | ]; cbn in Hs; try discriminate;
    injection Hs as _ Hen; subst en; destruct (lookup g cts); reflexivity.
Qed.

(* The slots that the index of a case says a set has, it has. *)
Lemma index_slots (fdecls : list (function_slot * code_id))
    (vs : list (value_slot * value))
    (idx : row_like_index set_of_closures_contents unit) :
  closures_index_satisfied fdecls vs idx ->
  (∀ g : function_slot,
     among g (contents_closures (closures_index_contents idx)) = true ->
     among g (domain fdecls) = true)
  ∧ ∀ w : value_slot,
      among w (contents_value_slots (closures_index_contents idx)) = true ->
      among w (domain vs) = true.
Proof.
  unfold closures_index_satisfied, closures_index_contents.
  destruct (index_domain idx) as [c | c]; intros [H1 H2];
    split; intros x Hx.
  - rewrite <- (H1 x). exact Hx.
  - rewrite <- (H2 x). exact Hx.
  - exact (proj1 (all_among _ _) H1 x Hx).
  - exact (proj1 (all_among _ _) H2 x Hx).
Qed.

(* A closure of a set has its own slot. *)
Lemma closure_in_set_slot (statics : static_env) (v : value)
    (slot : function_slot) (fdecls : list (function_slot * code_id))
    (vs : list (value_slot * value)) :
  closure_in_set statics v = Some (slot, fdecls, vs) ->
  is_bound slot fdecls = true.
Proof.
  destruct v as [c | code vs0 | slot0 fdecls0 vs0 | s | tag fields | elements
                | c | l | callee args arity];
    cbn [closure_in_set]; try discriminate.
  - intros [= <- <- <-]. reflexivity.
  - destruct (lookup slot0 fdecls0) eqn:Hl; cbn; [ | discriminate ].
    intros [= <- <- <-]. unfold is_bound. rewrite Hl. reflexivity.
  - destruct (lookup s (static_values statics)) as [[ | | | | | | ] | ];
      try discriminate.
    + intros [= <- <- <-]. reflexivity.
    + destruct (lookup slot0 function_decls) eqn:Hl; cbn; [ | discriminate ].
      intros [= <- <- <-]. unfold is_bound. rewrite Hl. reflexivity.
Qed.

(* A closure type that holds of [Null] and does not, of a value that is
   not [Null], hold with the nullability [Maybe_null None]. *)
Lemma maybe_null_of_not_null (statics : static_env) (vars : env)
    (v : value) (nn : or_unknown_or_bottom head_of_kind_value_non_null) :
  type_satisfied statics vars v
    (Type_value (Ok (No_alias (Head_of_kind_value nn Not_null)))) ->
  type_satisfied statics vars v
    (Type_value (Ok (No_alias (Head_of_kind_value nn (Maybe_null None))))).
Proof.
  rewrite !type_satisfied_value. cbn [descr_satisfied].
  rewrite !head_of_kind_value_satisfied_eq.
  intros [[_ []] | [Hnn [Hh _]]]. right.
  split; [ exact Hnn | split; [ exact Hh | exact I ] ].
Qed.

(* [slots_of_local_set], with the function that it computes the value
   slots with. *)
Definition local_value_slots (ctx : rewrite_context)
    (nodes : list (function_slot * node))
    : list value_slot -> option (list (value_slot * rewrite_metadata)) :=
  fix value_slots (ws : list value_slot)
      : option (list (value_slot * rewrite_metadata)) :=
    match ws with
    | [] => Some []
    | w :: ws =>
      match follow_field_for_set_of_closures ctx nodes w, value_slots ws with
      | None, _ | _, None => None
      | Some At_least_one_source_no_usages, Some rest => Some rest
      | Some m, Some rest => Some ((w, m) :: rest)
      end
    end.

Lemma slots_of_local_set_shape (ctx : rewrite_context)
    (vts : list value_slot) (fts : list function_slot)
    (set : set_of_closures) :
  ∃ fsl : list (function_slot * (rewrite_metadata * bool)),
    slots_of_local_set ctx vts fts set
    = match
        local_value_slots ctx (map (fun p => (fst p, fst (snd p))) set) vts
      with
      | None => None
      | Some vs => Some (vs, fsl)
      end
    ∧ map fst fsl = fts.
Proof.
  eexists. split; [ reflexivity | ].
  rewrite map_map. cbn [fst]. apply map_id.
Qed.

Lemma local_value_slots_some (ctx : rewrite_context)
    (nodes : list (function_slot * node)) (vts : list value_slot) :
  (∀ w : value_slot,
     w ∈ vts -> follow_field_for_set_of_closures ctx nodes w <> None) ->
  ∃ vs, local_value_slots ctx nodes vts = Some vs.
Proof.
  induction vts as [ | w vts IH]; intros H; [ eexists; reflexivity | ].
  cbn [local_value_slots].
  fold (local_value_slots ctx nodes).
  destruct (IH (fun w' Hw' => H w' (or_intror Hw'))) as [vs Hvs].
  rewrite Hvs.
  destruct (follow_field_for_set_of_closures ctx nodes w) as [m | ] eqn:Hm;
    [ | exfalso; exact (H w (or_introl eq_refl) Hm) ].
  destruct m; eexists; reflexivity.
Qed.

Lemma local_value_slots_spec (ctx : rewrite_context)
    (nodes : list (function_slot * node)) (vts : list value_slot) :
  ∀ vs : list (value_slot * rewrite_metadata),
    local_value_slots ctx nodes vts = Some vs ->
    ∀ (w : value_slot) (m0 : rewrite_metadata),
      lookup w vs = Some m0 ->
      w ∈ vts ∧ follow_field_for_set_of_closures ctx nodes w = Some m0
      ∧ m0 <> At_least_one_source_no_usages.
Proof.
  induction vts as [ | w vts IH]; intros vs H w' m0 Hl.
  - cbn in H. injection H as <-. discriminate Hl.
  - cbn [local_value_slots] in H. fold (local_value_slots ctx nodes) in H.
    destruct (follow_field_for_set_of_closures ctx nodes w) as [m | ] eqn:Hm;
      [ | discriminate H ].
    destruct (local_value_slots ctx nodes vts) as [rest | ] eqn:Hrest;
      [ | destruct m; discriminate H ].
    assert (Hrs : lookup w' rest = Some m0 ->
                  w' ∈ w :: vts
                  ∧ follow_field_for_set_of_closures ctx nodes w' = Some m0
                  ∧ m0 <> At_least_one_source_no_usages).
    { intros Hl'. destruct (IH rest eq_refl w' m0 Hl') as (Hin & Hf & Hne).
      split; [ right; exact Hin | split; assumption ]. }
    destruct m; injection H as <-; try exact (Hrs Hl);
      cbn [lookup] in Hl; destruct (Pos.eqb_spec w' w) as [-> | Hne];
      try exact (Hrs Hl);
      injection Hl as <-;
      (split; [ left; reflexivity | split; [ exact Hm | discriminate ] ]).
Qed.

(* [follow_field_for_set_of_closures] forgets only if no closure of the
   set uses the value slot. *)
Lemma follow_field_for_set_cases (ctx : rewrite_context)
    (nodes : list (function_slot * node)) (w : value_slot) :
  follow_field_for_set_of_closures ctx nodes w
  = Some At_least_one_source_no_usages
  ∨ ∃ (g : function_slot) (c : node),
      (g, c) ∈ nodes
      ∧ s_field_used (solution_of_tables (rc_tables ctx)) c
          (Field_value_slot w)
        = true.
Proof.
  unfold follow_field_for_set_of_closures. cbv zeta.
  destruct (forallb
              (fun p => negb (field_used ctx (snd p) (Field_value_slot w)))
              nodes) eqn:Hall; [ left; reflexivity | right ].
  assert (H : ∃ p : function_slot * node,
             p ∈ nodes ∧ field_used ctx (snd p) (Field_value_slot w) = true).
  { clear - Hall. induction nodes as [ | p nodes IH]; [ discriminate Hall | ].
    cbn [forallb] in Hall. apply andb_false_iff in Hall as [H | H].
    - exists p. split; [ left; reflexivity | ].
      apply negb_false_iff in H. exact H.
    - destruct (IH H) as (q & Hq & Hfu). exists q.
      split; [ right; exact Hq | exact Hfu ]. }
  destruct H as ([g c] & Hin & Hfu). exists g, c. split; [ exact Hin | ].
  exact Hfu.
Qed.

(* ======================================================================== *)
(* Null keeps its types                                                     *)
(* ======================================================================== *)

Section Null.

Variable fx : fixes.
Hypothesis Hfix_null : fix_null fx = true.

Variable ctx : rewrite_context.
Variable tenv : Typing_env.t.
Variable canon : canon_fn.
Variables statics statics' : static_env.
Variables valuation valuation' : env.

Hypothesis Henv : typing_env_satisfied_strictly statics valuation tenv.
Hypothesis Hcopies : copies_valued tenv canon valuation valuation'.

Local Abbreviation null := (Value_const Null).

Definition null_kept (rw : rewrite_metadata -> flambda_type -> flambda_type)
    : Prop :=
  ∀ (m : rewrite_metadata) (t : flambda_type),
    type_satisfied statics valuation null t ->
    type_satisfied statics' valuation' null (rw m t).

Lemma null_unknown (t : flambda_type) :
  type_satisfied statics valuation null t ->
  type_satisfied statics' valuation' null (unknown (kind t)).
Proof.
  intros Hsat. apply unknown_satisfied.
  exact (type_satisfied_kind _ _ _ _ Hsat).
Qed.

(* A head of kind value that [Null] has: it may be null, and so may the
   head rewritten. *)
Lemma null_head (rw : rewrite_metadata -> flambda_type -> flambda_type)
    (m : rewrite_metadata) (head : head_of_kind_value) :
  head_of_kind_value_satisfied statics valuation null head ->
  head_of_kind_value_satisfied statics' valuation' null
    (rewrite_head_of_kind_value fx ctx rw m head).
Proof.
  destruct head as [non_null is_null].
  rewrite !head_of_kind_value_satisfied_eq.
  intros [[_ His] | [Hnn _]]; [ | contradiction ].
  left. split; [ reflexivity | ]. destruct is_null; [ contradiction | ].
  exact I.
Qed.

Lemma null_identity (rw : rewrite_metadata -> flambda_type -> flambda_type)
    (m : rewrite_metadata) (t : flambda_type) :
  type_satisfied statics valuation null t ->
  type_satisfied statics' valuation' null
    (rewrite_identity fx ctx tenv rw m t).
Proof.
  intros Hsat. unfold rewrite_identity.
  destruct (expand_head tenv t) as [e | ] eqn:He;
    [ | exact (null_unknown t Hsat) ].
  pose proof (expand_head_sound statics valuation tenv t e null Henv Hsat He)
    as Hsat_e.
  assert (Hmain : type_satisfied statics' valuation' null
                    (rewrite_expanded_head fx ctx rw m e)).
  { destruct e as [k [ | h | ]]; unfold rewrite_expanded_head;
      cbn [Expanded_type.descr Expanded_type.kind Expanded_type.to_type] in *.
    - apply unknown_satisfied. exact (proj1 (unknown_satisfied _ _ _ _) Hsat_e).
    - destruct h as [hv | hi | kn ns | hr | hreg].
      + rewrite type_satisfied_value in Hsat_e |- *.
        exact (null_head rw m hv Hsat_e).
      + rewrite type_satisfied_naked_immediate in Hsat_e.
        cbn [descr_satisfied] in Hsat_e.
        destruct Hsat_e as (i & Hi & _). discriminate Hi.
      + rewrite type_satisfied_naked_number in Hsat_e.
        cbn [descr_satisfied] in Hsat_e.
        destruct Hsat_e as (c & n & [= <-] & Hk & _). discriminate Hk.
      + rewrite type_satisfied_rec_info in Hsat_e.
        cbn [descr_satisfied] in Hsat_e. discriminate Hsat_e.
      + rewrite type_satisfied_region in Hsat_e.
        cbn [descr_satisfied] in Hsat_e. discriminate Hsat_e.
    - exfalso. exact (bottom_satisfied _ _ _ _ Hsat_e). }
  destruct (must_be_singleton (rewrite_expanded_head fx ctx rw m e))
    as [[c | ] | ] eqn:Hmb; try exact Hmain.
  pose proof (must_be_singleton_satisfied _ _ null _ c Hmb Hmain) as Hc.
  rewrite Hc. apply alias_satisfied.
  split; [ reflexivity | ]. rewrite <- Hc.
  exact (type_satisfied_kind _ _ _ _ Hmain).
Qed.

(* The rule of a closure says that the closure may be null where it is
   told so. *)
Lemma closure_rule_maybe_null (fx' : fixes) (ctx' : rewrite_context)
    (usages : rewrite_metadata) (slot : function_slot)
    (alloc_mode : Alloc_mode.t) (entry : closures_entry) (maybe_null : bool)
    (spec : closure_rewrite) :
  closure_rule fx' ctx' usages slot alloc_mode entry maybe_null
  = Rule_closure spec ->
  cr_maybe_null spec = maybe_null.
Proof.
  unfold closure_rule. destruct entry as [fts cts vsts].
  repeat match goal with
         | |- context [match ?x with pair _ _ => _ end] => destruct x
         | |- context [if ?b then _ else _] => destruct b
         | |- context [match ?l with nil => _ | cons _ _ => _ end] =>
           destruct l
         end;
    intros H; try discriminate H; injection H as <-; reflexivity.
Qed.

(* Where the rule of a closure applies to a type that [Null] has, the head
   of the type may be null. *)
Lemma null_closure_rule (m : rewrite_metadata) (t : flambda_type)
    (spec : closure_rewrite) :
  type_satisfied statics valuation null t ->
  rewrite_rule fx ctx tenv m t = Rule_closure spec ->
  cr_maybe_null spec = true.
Proof.
  intros Hsat. unfold rewrite_rule. destruct (forgets m); [ discriminate | ].
  destruct (kind t); try discriminate.
  destruct (Provers.meet_single_closures_entry tenv t)
    as [[[[[slot am] entry] ft] | | ] | ] eqn:Hm; try discriminate.
  unfold Provers.meet_single_closures_entry, Provers.gen_value_to_meet,
    Provers.expand_head in Hm.
  intros Hspec.
  assert (Hmn : expanded_maybe_null tenv t = true).
  { unfold expanded_maybe_null.
    destruct (expand_head tenv t) as [e | ] eqn:He; cbn in Hm;
      [ | discriminate ].
    pose proof (expand_head_sound statics valuation tenv t e null Henv Hsat
                  He) as Hsat_e.
    destruct e as [k [ | [[non_null is_null] | | | | ] | ]];
      unfold descr_oub_of_expanded in Hm;
      cbn [Expanded_type.descr Expanded_type.kind] in Hm |- *;
      try (destruct k as [ | [] | | ]; discriminate);
      try discriminate.
    change (type_satisfied statics valuation null
              (Type_value (Ok (No_alias (Head_of_kind_value non_null
                                           is_null)))))
      in Hsat_e.
    rewrite type_satisfied_value in Hsat_e. cbn [descr_satisfied] in Hsat_e.
    rewrite head_of_kind_value_satisfied_eq in Hsat_e.
    destruct Hsat_e as [[_ His] | [Hnn _]]; [ | contradiction ].
    destruct is_null; [ contradiction | reflexivity ]. }
  rewrite (closure_rule_maybe_null _ _ _ _ _ _ _ _ Hspec). exact Hmn.
Qed.

Lemma null_closure (rw : rewrite_metadata -> flambda_type -> flambda_type)
    (spec : closure_rewrite) (t : flambda_type) :
  cr_maybe_null spec = true ->
  type_satisfied statics' valuation' null (rewrite_closure fx tenv rw spec t).
Proof.
  intros Hmn. unfold rewrite_closure. rewrite Hfix_null, Hmn. cbn [andb].
  unfold at_least_this_closure, create_closures, non_null_value.
  rewrite type_satisfied_value. cbn [descr_satisfied].
  rewrite head_of_kind_value_satisfied_eq. left.
  split; [ reflexivity | exact I ].
Qed.

Lemma rewrite_keeps_null_fuel (fuel : nat) :
  null_kept (rewrite_arbitrary fx ctx tenv canon fuel)
  ∧ null_kept (rewrite fx ctx tenv canon fuel).
Proof.
  induction fuel as [ | fuel [IHa IHr]]; split; intros m t Hsat.
  - exact (null_unknown t Hsat).
  - exact (null_unknown t Hsat).
  - cbn [rewrite_arbitrary].
    destruct (get_alias_opt t) as [s | ] eqn:Ha; [ | exact (IHr m t Hsat) ].
    pose proof (proj1 (get_alias_satisfied _ _ null t s Ha) Hsat) as [Hs Hkt].
    destruct (Typing_env.get_canonical_simple_exn tenv
                (Some Name_mode.in_types) None s)
      as [[[x | sym | c] | ] | ] eqn:Hcan;
      try exact (null_unknown t Hsat).
    + destruct (fix_alias fx && forgets m); [ exact (null_unknown t Hsat) | ].
      apply alias_satisfied. split; [ | exact Hkt ].
      change (value_of_name valuation' (canon (Name_var x) (kind t) m)
              = Some null).
      rewrite (Hcopies s (Name_var x) (kind t) m Hcan). exact Hs.
    + destruct (fix_alias fx && forgets m); [ exact (null_unknown t Hsat) | ].
      apply alias_satisfied. split; [ | exact Hkt ].
      change (value_of_name valuation' (canon (Name_symbol sym) (kind t) m)
              = Some null).
      rewrite (Hcopies s (Name_symbol sym) (kind t) m Hcan). exact Hs.
    + pose proof (get_canonical_simple_satisfied _ _ _ _ _ _ _ _ Henv Hcan)
        as Hceq.
      rewrite Hs in Hceq. cbn in Hceq. injection Hceq as Hc.
      rewrite Hc. apply IHr. apply type_for_const_satisfied.
  - cbn [rewrite].
    destruct (rewrite_rule fx ctx tenv m t) as [ | | spec] eqn:Hrule.
    + exact (null_identity (rewrite_arbitrary fx ctx tenv canon fuel) m t
               Hsat).
    + exact (null_unknown t Hsat).
    + exact (null_closure (rewrite_arbitrary fx ctx tenv canon fuel) spec t
               (null_closure_rule m t spec Hsat Hrule)).
Qed.

End Null.

(* ======================================================================== *)
(* The rewriting is sound                                                   *)
(* ======================================================================== *)

Section Sound.

Variable fx : fixes.
Hypothesis Hfix_boxed : fix_boxed fx = true.
Hypothesis Hfix_alias : fix_alias fx = true.
Hypothesis Hfix_refine : fix_refine fx = true.
Hypothesis Hfix_null : fix_null fx = true.

Variable is_local : field -> bool.
Variable F : facts.
Variable T : tables.
Variable sets : list set_of_closures.
Variable tenv : Typing_env.t.
Variable canon : canon_fn.
Variables statics statics' : static_env.
Variables valuation valuation' : env.

Local Abbreviation ctx := (plain_context is_local F T sets).
Local Abbreviation G := (graph_of_facts F).
Local Abbreviation sol := (solution_of_tables T).

Hypothesis Hc : closed is_local G sol.
Hypothesis Hleast : least_solution is_local F T.
Hypothesis Hlocal : is_local_slots_only is_local.
Hypothesis Hsets : sets_of_graph G sets.
Hypothesis Henv : typing_env_satisfied_strictly statics valuation tenv.
Hypothesis Hfrag : env_in_fragment valuation tenv.
Hypothesis Halloc : statics_allocated sol statics.
Hypothesis Hst : statics_rebuilt G sol statics statics'.
Hypothesis Hcopies : copies_valued tenv canon valuation valuation'.

Lemma not_local_block (i : nat) (k : flambda_kind) :
  is_local (Field_block i k) = false.
Proof.
  destruct (is_local (Field_block i k)) eqn:H; [ | reflexivity ].
  destruct (Hlocal _ H) as [(w & Hw) | (g & Hg)]; discriminate.
Qed.

Lemma not_local_boxed (bn : boxable_number) :
  is_local (Field_boxed_number bn) = false.
Proof.
  destruct (is_local (Field_boxed_number bn)) eqn:H; [ | reflexivity ].
  destruct (Hlocal _ H) as [(w & Hw) | (g & Hg)]; discriminate.
Qed.

(* What a rewriting of the arbitrary types is asked: to be sound, and to
   give, under a metadata that forgets, a type that every value of the
   kind has. *)
Definition rw_sound (rw : rewrite_metadata -> flambda_type -> flambda_type)
    : Prop :=
  ∀ (m : rewrite_metadata) (t : flambda_type) (v v' : value),
    type_in_fragment valuation t -> rebuilt_under G sol m v v' ->
    type_satisfied statics valuation v t ->
    type_satisfied statics' valuation' v' (rw m t).

Definition rw_forgets (rw : rewrite_metadata -> flambda_type -> flambda_type)
    : Prop :=
  ∀ (m : rewrite_metadata) (t : flambda_type) (v v' : value),
    forgets m = true -> type_in_fragment valuation t ->
    type_satisfied statics valuation v t ->
    kind_of_value v' = kind_of_value v ->
    type_satisfied statics' valuation' v' (rw m t).

Lemma sat_unknown_of_kind (v v' : value) (t : flambda_type) :
  type_satisfied statics valuation v t ->
  kind_of_value v' = kind_of_value v ->
  type_satisfied statics' valuation' v' (unknown (kind t)).
Proof.
  intros Hv Hk. apply unknown_satisfied. rewrite Hk.
  exact (type_satisfied_kind _ _ _ _ Hv).
Qed.

Lemma canonical_eval (s c : simple) :
  Typing_env.get_canonical_simple_exn tenv (Some Name_mode.in_types) None s
  = Some (Some c) ->
  eval_simple valuation c = eval_simple valuation s.
Proof.
  intros H. exact (get_canonical_simple_satisfied _ _ _ _ _ _ _ _ Henv H).
Qed.

Lemma rule_forgets (m : rewrite_metadata) (t : flambda_type) :
  forgets m = true -> rewrite_rule fx ctx tenv m t = Rule_forget.
Proof. intros H. unfold rewrite_rule. rewrite H. reflexivity. Qed.

Lemma rule_identity_not_forgets (m : rewrite_metadata) (t : flambda_type) :
  rewrite_rule fx ctx tenv m t = Rule_identity -> forgets m = false.
Proof.
  unfold rewrite_rule. destruct (forgets m); [ discriminate | reflexivity ].
Qed.

(* Under a metadata that forgets. *)
Lemma rewrite_forgets (fuel : nat) :
  rw_forgets (rewrite_arbitrary fx ctx tenv canon fuel)
  ∧ rw_forgets (rewrite fx ctx tenv canon fuel).
Proof.
  induction fuel as [ | fuel [IHa IHr]]; split; intros m t v v' Hm Ht Hv Hk.
  - exact (sat_unknown_of_kind v v' t Hv Hk).
  - exact (sat_unknown_of_kind v v' t Hv Hk).
  - cbn [rewrite_arbitrary].
    destruct (get_alias_opt t) as [s | ] eqn:Ha;
      [ | exact (IHr m t v v' Hm Ht Hv Hk) ].
    pose proof (proj1 (get_alias_satisfied _ _ v t s Ha) Hv) as [Hs Hkt].
    destruct (Typing_env.get_canonical_simple_exn tenv
                (Some Name_mode.in_types) None s)
      as [[[x | sym | c] | ] | ] eqn:Hcan;
      try (rewrite Hfix_alias, Hm; exact (sat_unknown_of_kind v v' t Hv Hk));
      try exact (sat_unknown_of_kind v v' t Hv Hk).
    pose proof (canonical_eval s (Const c) Hcan) as Hceq.
    rewrite Hs in Hceq. cbn in Hceq. injection Hceq as <-.
    apply (IHr m (type_for_const c) (Value_const c) v' Hm).
    + apply type_for_const_fragment.
    + apply type_for_const_satisfied.
    + exact Hk.
  - cbn [rewrite]. rewrite (rule_forgets m t Hm).
    exact (sat_unknown_of_kind v v' t Hv Hk).
Qed.

(* A block of the first run, and what the second has for it. *)
Lemma block_rebuilt (m : rewrite_metadata) (v v' : value) (tag : Z)
    (fields : list value) :
  forgets m = false -> rebuilt_under G sol m v v' ->
  block_of statics v = Some (tag, fields) ->
  ∃ (a : node) (fields' : list value),
    describes sol m a ∧ block_of statics' v' = Some (tag, fields')
    ∧ fields_rebuilt (field_rebuilt G sol (rebuilt_value G sol) a) 0 fields
        fields'.
Proof.
  intros Hm [(c & -> & ->) | (n & Hv & Hat & Hhu)] Hblock;
    [ discriminate Hblock | ].
  specialize (Hhu Hm).
  destruct v as [c | code vs | slot decls vs | s | tag0 fields0 | elements | c
                | l | callee args arity];
    cbn [block_of] in Hblock; try discriminate.
  - destruct (lookup s (static_values statics)) as [allocated | ] eqn:Hl;
      [ | discriminate ].
    destruct allocated; try discriminate. injection Hblock as <- <-.
    cbn [rebuilt_value] in Hv. destruct Hv as (-> & Hmcf & Hhus).
    pose proof (Hst s _ Hl (Hhus Hhu)) as Hsr. cbn [static_rebuilt] in Hsr.
    destruct Hsr as (fields' & Hl' & Hfields).
    exists (Node_symbol s), fields'. split.
    + apply (Hat (Node_symbol s) Hmcf).
      destruct (Halloc s _ Hl) as (Hhs & _). exact Hhs.
    + split; [ cbn [block_of]; rewrite Hl'; reflexivity | exact Hfields ].
  - injection Hblock as <- <-. cbn [rebuilt_value] in Hv.
    destruct Hv as (a & fields' & Hmcf & Hhs & -> & Hfields).
    exists a, fields'.
    split; [ exact (Hat a Hmcf Hhs) | split; [ reflexivity | exact Hfields ] ].
Qed.

(* A boxed number of the first run, whose contents the metadata does not
   forget: the second has the same. *)
Lemma boxed_rebuilt (m : rewrite_metadata) (v v' : value)
    (k : boxable_number) (c : const) :
  forgets m = false -> rebuilt_under G sol m v v' ->
  boxed_number_of statics v = Some c -> kind_of_const c = unboxed_kind k ->
  forgets (follow_field ctx m (Field_boxed_number k)) = false ->
  boxed_number_of statics' v' = Some c.
Proof.
  intros Hm [(c0 & -> & ->) | (n & Hv & Hat & Hhu)] Hbox Hkc Hff;
    [ discriminate Hbox | ].
  specialize (Hhu Hm).
  destruct v as [c0 | code vs | slot decls vs | s | tag0 fields0 | elements
                | c0 | l | callee args arity];
    cbn [boxed_number_of] in Hbox; try discriminate.
  - destruct (lookup s (static_values statics)) as [allocated | ] eqn:Hl;
      [ | discriminate ].
    cbn [rebuilt_value] in Hv. destruct Hv as (-> & Hmcf & Hhus).
    pose proof (Hst s allocated Hl (Hhus Hhu)) as Hsr.
    destruct allocated as [ | | | contents | | | ]; try discriminate.
    cbn [static_rebuilt] in Hsr. cbn [boxed_number_of]. rewrite Hsr.
    exact Hbox.
  - cbn [rebuilt_value] in Hv.
    destruct Hv as (a & bn & from & Hmcf & Hhs & Hbn & Hcons & ->).
    pose proof (boxable_kind_eq bn k c0 c Hbn Hbox Hkc) as <-.
    destruct (s_field_used sol a (Field_boxed_number bn)) eqn:Hfu;
      [ exact Hbox | exfalso ].
    rewrite (follow_field_forgets ctx is_local Hc m a from _ (Hat a Hmcf Hhs)
               Hm Hcons (not_local_boxed bn) Hfu) in Hff.
    discriminate Hff.
Qed.

(* A string of the first run is one of the second. *)
Lemma string_rebuilt (m : rewrite_metadata) (v v' : value) (bytes : list Z) :
  forgets m = false -> rebuilt_under G sol m v v' ->
  string_of statics v = Some bytes -> string_of statics' v' = Some bytes.
Proof.
  intros Hm [(c & -> & ->) | (n & Hv & Hat & Hhu)] Hstr;
    [ discriminate Hstr | ].
  specialize (Hhu Hm).
  destruct v as [c | code vs | slot decls vs | s | tag0 fields0 | elements | c
                | l | callee args arity];
    cbn [string_of] in Hstr; try discriminate.
  destruct (lookup s (static_values statics)) as [allocated | ] eqn:Hl;
    [ | discriminate ].
  destruct allocated; try discriminate. injection Hstr as <-.
  cbn [rebuilt_value] in Hv. destruct Hv as (-> & _ & Hhus).
  pose proof (Hst s _ Hl (Hhus Hhu)) as Hsr. cbn [static_rebuilt] in Hsr.
  cbn [string_of]. rewrite Hsr. reflexivity.
Qed.

Section With_rw.

Variable rw : rewrite_metadata -> flambda_type -> flambda_type.
Hypothesis Hrw : rw_sound rw.
Hypothesis Hrwf : rw_forgets rw.

(* A field of a block. *)
Lemma field_sound (m : rewrite_metadata) (a : node) (k : nat)
    (field field' : value) (ty : flambda_type) :
  forgets m = false -> describes sol m a ->
  field_rebuilt G sol (rebuilt_value G sol) a k field field' ->
  type_in_fragment valuation ty ->
  type_satisfied statics valuation field ty ->
  type_satisfied statics' valuation' field' (rw (block_slot ctx m k ty) ty).
Proof.
  intros Hm Hd (from & Hcons & Hused & Hunused) Hfr Hsat.
  unfold block_slot. rewrite <- (type_satisfied_kind _ _ _ _ Hsat).
  destruct (s_field_used sol a (Field_block k (kind_of_value field)))
    eqn:Hfu.
  - destruct (Hused eq_refl) as [Hhu Hrv].
    apply (Hrw _ ty field field' Hfr); [ | exact Hsat ].
    right. exists from. split; [ exact Hrv | split; [ | intros _; exact Hhu ] ].
    exact (follow_field_at ctx is_local Hc m a from _ Hd Hm Hcons
             (not_local_block k _) Hfu).
  - rewrite (Hunused eq_refl).
    apply (Hrwf _ ty field _
             (follow_field_forgets ctx is_local Hc m a from _ Hd Hm Hcons
                (not_local_block k _) Hfu)
             Hfr Hsat).
    reflexivity.
Qed.

(* The fields of a block, from the [k]th. *)
Lemma fields_sound (m : rewrite_metadata) (a : node) :
  forgets m = false -> describes sol m a ->
  ∀ (tys : list flambda_type) (k : nat) (fields fields' : list value),
    fields_rebuilt (field_rebuilt G sol (rebuilt_value G sol) a) k fields
      fields' ->
    (∀ ty : flambda_type, ty ∈ tys -> type_in_fragment valuation ty) ->
    fields_satisfied (type_satisfied statics valuation) tys fields ->
    fields_satisfied (type_satisfied statics' valuation')
      (map (fun p => rw (block_slot ctx m (fst p) (snd p)) (snd p))
         (combine (seq k (length tys)) tys))
      fields'.
Proof.
  intros Hm Hd tys. induction tys as [ | ty tys IH];
    intros k fields fields' Hfields Hfr Hfs; [ exact I | ].
  destruct fields as [ | field fields]; destruct fields' as [ | field' fields'];
    cbn [fields_rebuilt] in Hfields; try contradiction.
  - cbn. exact I.
  - destruct Hfields as [Hfield Hfields]. destruct Hfs as [Hf Hfs].
    cbn [length seq combine map fst snd]. split.
    + exact (field_sound m a k field field' ty Hm Hd Hfield
               (Hfr ty (or_introl eq_refl)) Hf).
    + exact (IH (S k) fields fields' Hfields
               (fun ty' H => Hfr ty' (or_intror H)) Hfs).
Qed.

(* A case of blocks. *)
Lemma block_case_sound (m : rewrite_metadata) (a : node) (tag : Z)
    (fields fields' : list value) (case : row_like_block_case) :
  forgets m = false -> describes sol m a ->
  fields_rebuilt (field_rebuilt G sol (rebuilt_value G sol) a) 0 fields
    fields' ->
  (∀ ty : flambda_type, ty ∈ case_maps_to case -> type_in_fragment valuation ty)
  ->
  row_like_block_case_satisfied statics valuation tag fields case ->
  row_like_block_case_satisfied statics' valuation' tag fields'
    (rewrite_block_case ctx rw m case).
Proof.
  intros Hm Hd Hfields Hfr [Hidx [Hfs _]].
  destruct case as [tys idx ext].
  cbn [rewrite_block_case case_maps_to case_index case_env_extension] in *.
  split; [ | split; [ | apply empty_env_extension_satisfied ] ].
  - unfold block_index_satisfied, block_has_shape in *.
    rewrite (fields_rebuilt_length _ 0 fields fields' Hfields).
    rewrite (fields_rebuilt_kinds G sol a 0 fields fields' Hfields).
    exact Hidx.
  - unfold rewrite_fields.
    exact (fields_sound m a Hm Hd tys 0 fields fields' Hfields Hfr Hfs).
Qed.

(* The row-like structure of blocks. *)
Lemma blocks_sound (m : rewrite_metadata) (a : node) (tag : Z)
    (fields fields' : list value) (blocks : row_like_for_blocks) :
  forgets m = false -> describes sol m a ->
  fields_rebuilt (field_rebuilt G sol (rebuilt_value G sol) a) 0 fields
    fields' ->
  shaped_blocks blocks = true
  ∧ (∀ (s : simple) (k : flambda_kind) (w : value),
       (s, k) ∈ aliases_of_row_like_for_blocks blocks ->
       eval_simple valuation s = Some w -> unallocated w) ->
  row_like_for_blocks_satisfied statics valuation tag fields blocks ->
  row_like_for_blocks_satisfied statics' valuation' tag fields'
    (rewrite_row_like_for_blocks ctx rw m blocks).
Proof.
  intros Hm Hd Hfields [Hcf Hal] Hsat.
  destruct blocks as [known other am].
  cbn [rewrite_row_like_for_blocks].
  rewrite row_like_for_blocks_find_iff in Hsat |- *.
  rewrite keyed_lookup_map_values.
  match goal with
  | |- context [option_map _ ?l] =>
    destruct l as [[case | ] | ] eqn:Hl
  end;
    cbn [option_map].
  - apply (block_case_sound m a tag fields fields' case Hm Hd Hfields);
      [ | exact Hsat ].
    apply (known_case_fragment valuation known other am tag case Hcf Hal).
    exact (keyed_lookup_in _ Z.eqb_eq tag _ known Hl).
  - exact I.
  - destruct other as [case | ]; [ | contradiction ].
    exact (block_case_sound m a tag fields fields' case Hm Hd Hfields
             (other_case_fragment valuation known case am Hcf Hal) Hsat).
Qed.

(* A head of kind value that is not null. *)
Lemma non_null_sound (m : rewrite_metadata)
    (head : head_of_kind_value_non_null) (v v' : value) :
  forgets m = false -> non_null_in_fragment valuation head ->
  rebuilt_under G sol m v v' -> v <> Value_const Null ->
  head_of_kind_value_non_null_satisfied statics valuation v head ->
  match rewrite_head_of_kind_value_non_null fx ctx rw m head with
  | Unknown => kind_of_value v' = Value
  | Bottom => False
  | Ok head' => head_of_kind_value_non_null_satisfied statics' valuation' v'
                  head'
  end.
Proof.
  intros Hm Hfr Hv Hnn Hsat.
  destruct head as [is_int immediates get_tag blocks extensions is_unique
                   | am | k contents am | closures am | strs].
  - cbn [rewrite_head_of_kind_value_non_null].
    rewrite variant_satisfied in Hsat |- *.
    destruct Hsat as [(i & -> & _ & Himm & _)
                     | (tag & fields & Hblock & _ & _ & Hblocks & _)].
    + left. rewrite (rebuilt_under_const _ _ m _ v' Hv). exists i.
      split; [ reflexivity | split; [ exact I | split; [ | exact I ] ] ].
      destruct immediates as [ty | ]; [ | exact I ].
      apply (Hrw m ty (Value_const (Naked_immediate i))
               (Value_const (Naked_immediate i))).
      * exact (immediates_fragment valuation is_int ty get_tag blocks
                 extensions is_unique Hfr).
      * left. exists (Naked_immediate i). split; reflexivity.
      * exact Himm.
    + destruct (block_rebuilt m v v' tag fields Hm Hv Hblock)
        as (a & fields' & Hd & Hblock' & Hfields).
      right. exists tag, fields'.
      split; [ exact Hblock' | split; [ exact I | split; [ exact I | ] ] ].
      split; [ | exact I ].
      destruct blocks as [blocks | ]; [ | exact I ].
      exact (blocks_sound m a tag fields fields' blocks Hm Hd Hfields
               (blocks_fragment valuation is_int immediates get_tag blocks
                  extensions is_unique Hfr)
               Hblocks).
  - rewrite mutable_block_head_satisfied in Hsat. contradiction.
  - cbn [rewrite_head_of_kind_value_non_null]. rewrite Hfix_boxed.
    rewrite boxed_number_satisfied in Hsat.
    destruct Hsat as (c & Hbox & Hkc & Hcontents).
    destruct (forgets (follow_field ctx m (Field_boxed_number k))) eqn:Hff;
      cbn [andb].
    + rewrite (rebuilt_under_kind _ _ m v v' Hv).
      exact (boxed_number_of_kind statics v c Hbox).
    + rewrite boxed_number_satisfied. exists c.
      split; [ exact (boxed_rebuilt m v v' k c Hm Hv Hbox Hkc Hff) | ].
      split; [ exact Hkc | ].
      apply (Hrw m contents (Value_const c) (Value_const c)
               (contents_fragment valuation k contents am Hfr));
        [ left; exists c; split; reflexivity | exact Hcontents ].
  - cbn [rewrite_head_of_kind_value_non_null].
    destruct closures as [[ | b known]].
    + rewrite closures_satisfied in Hsat.
      destruct Hsat as (slot & fdecls & vs & _ & Hrow).
      rewrite row_like_for_closures_lookup_iff in Hrow.
      destruct Hrow as (case & Hl & _). discriminate Hl.
    + rewrite (rebuilt_under_kind _ _ m v v' Hv).
      exact (proj1 (head_non_null_kind _ _ _ _ Hsat)).
  - cbn [rewrite_head_of_kind_value_non_null].
    rewrite string_satisfied in Hsat |- *.
    destruct Hsat as (bytes & Hstr & Hin). exists bytes.
    split; [ exact (string_rebuilt m v v' bytes Hm Hv Hstr) | exact Hin ].
Qed.

(* A head of kind value. *)
Lemma head_value_sound (m : rewrite_metadata) (head : head_of_kind_value)
    (v v' : value) :
  forgets m = false ->
  type_in_fragment valuation (Type_value (Ok (No_alias head))) ->
  rebuilt_under G sol m v v' ->
  head_of_kind_value_satisfied statics valuation v head ->
  head_of_kind_value_satisfied statics' valuation' v'
    (rewrite_head_of_kind_value fx ctx rw m head).
Proof.
  intros Hm Hfr Hv Hsat. destruct head as [non_null is_null].
  cbn [rewrite_head_of_kind_value].
  rewrite head_of_kind_value_satisfied_eq in Hsat |- *.
  destruct Hsat as [[Hnull His] | [Hnn [Hpart His]]].
  - left. subst v. rewrite (rebuilt_under_const _ _ m Null v' Hv).
    split; [ reflexivity | ].
    destruct is_null; [ contradiction | exact I ].
  - right. split; [ exact (rebuilt_under_not_null _ _ m v v' Hv Hnn) | ].
    split; [ | destruct is_null; exact I ].
    destruct non_null as [ | h | ]; [ | | contradiction ].
    + rewrite (rebuilt_under_kind _ _ m v v' Hv). exact Hpart.
    + exact (non_null_sound m h v v' Hm (head_fragment valuation h is_null Hfr)
               Hv Hnn Hpart).
Qed.

(* An expanded type. *)
Lemma expanded_sound (m : rewrite_metadata) (e : Expanded_type.t)
    (v v' : value) :
  forgets m = false ->
  type_in_fragment valuation (Expanded_type.to_type e) ->
  rebuilt_under G sol m v v' ->
  type_satisfied statics valuation v (Expanded_type.to_type e) ->
  type_satisfied statics' valuation' v' (rewrite_expanded_head fx ctx rw m e).
Proof.
  intros Hm Hfr Hv Hsat.
  destruct e as [k [ | h | ]]; unfold rewrite_expanded_head;
    cbn [Expanded_type.descr Expanded_type.kind Expanded_type.to_type] in *.
  - apply unknown_satisfied. rewrite (rebuilt_under_kind _ _ m v v' Hv).
    exact (proj1 (unknown_satisfied _ _ _ _) Hsat).
  - destruct h as [hv | hi | kn ns | hr | hreg].
    + rewrite type_satisfied_value in Hsat |- *.
      exact (head_value_sound m hv v v' Hm Hfr Hv Hsat).
    + rewrite type_satisfied_naked_immediate in Hsat |- *.
      cbn [descr_satisfied] in Hsat |- *.
      destruct Hsat as (i & -> & Hnum & _).
      rewrite (rebuilt_under_const _ _ m _ v' Hv). exists i.
      split; [ reflexivity | split; [ exact Hnum | ] ].
      intros relation n [].
    + rewrite type_satisfied_naked_number in Hsat |- *.
      cbn [descr_satisfied] in Hsat |- *.
      destruct Hsat as (c & n & -> & Hk & Hn & Hin).
      rewrite (rebuilt_under_const _ _ m _ v' Hv). exists c, n. auto.
    + rewrite type_satisfied_rec_info in Hsat |- *.
      cbn [descr_satisfied] in Hsat |- *.
      rewrite (rebuilt_under_kind _ _ m v v' Hv). exact Hsat.
    + rewrite type_satisfied_region in Hsat |- *.
      cbn [descr_satisfied] in Hsat |- *.
      rewrite (rebuilt_under_kind _ _ m v v' Hv). exact Hsat.
  - exfalso. exact (bottom_satisfied _ _ _ _ Hsat).
Qed.

(* A type rewritten under [Rule.identity]. *)
Lemma identity_sound (m : rewrite_metadata) (t : flambda_type) (v v' : value)
    :
  forgets m = false -> type_in_fragment valuation t ->
  rebuilt_under G sol m v v' -> type_satisfied statics valuation v t ->
  type_satisfied statics' valuation' v' (rewrite_identity fx ctx tenv rw m t).
Proof.
  intros Hm Ht Hv Hsat. unfold rewrite_identity.
  destruct (expand_head tenv t) as [e | ] eqn:He;
    [ | exact (sat_unknown_of_kind v v' t Hsat
                 (rebuilt_under_kind _ _ m v v' Hv)) ].
  pose proof (expand_head_sound statics valuation tenv t e v Henv Hsat He)
    as Hsat_e.
  pose proof (expand_head_fragment valuation tenv Hfrag t e He Ht) as Hfr_e.
  pose proof (expanded_sound m e v v' Hm Hfr_e Hv Hsat_e) as Hmain.
  destruct (must_be_singleton (rewrite_expanded_head fx ctx rw m e))
    as [[c | ] | ] eqn:Hmb; try exact Hmain.
  pose proof (must_be_singleton_satisfied _ _ v' _ c Hmb Hmain) as ->.
  apply alias_satisfied.
  split; [ reflexivity | exact (type_satisfied_kind _ _ _ _ Hmain) ].
Qed.

End With_rw.

(* ------------------------------------------------------------------------ *)
(* Closures                                                                 *)
(* ------------------------------------------------------------------------ *)

(* The closure that a closure type holds of, read through the head the
   type is expanded to: [Null], or a closure at the slot of the single
   case, which the case admits. *)
Lemma closure_of_type (t : flambda_type) (e : Expanded_type.t)
    (known : finite_map row_like_closures_case) (am : Alloc_mode.t)
    (isn : is_null) (slot : function_slot) (entry : closures_entry)
    (v : value) :
  expand_head tenv t = Some e ->
  Expanded_type.descr e
  = Ok (Expanded_type.Head_value
          (Head_of_kind_value
             (Ok (Closures (Row_like_for_closures known) am)) isn)) ->
  single_closures_entry
    (row_like_for_closures_get_single_tag (Row_like_for_closures known))
  = Some (slot, entry) ->
  type_satisfied statics valuation v t ->
  v = Value_const Null
  ∨ ∃ (fdecls : list (function_slot * code_id))
      (vs : list (value_slot * value)) (case : row_like_closures_case),
      closure_in_set statics v = Some (slot, fdecls, vs)
      ∧ lookup slot known = Some case ∧ case_maps_to case = entry
      ∧ row_like_closures_case_satisfied statics valuation v slot fdecls vs
          case.
Proof.
  intros He Hd Hs Hsat.
  pose proof (expand_head_sound statics valuation tenv t e v Henv Hsat He)
    as Hsat_e.
  unfold Expanded_type.to_type in Hsat_e. rewrite Hd in Hsat_e.
  rewrite type_satisfied_value in Hsat_e. cbn [descr_satisfied] in Hsat_e.
  rewrite head_of_kind_value_satisfied_eq in Hsat_e.
  destruct Hsat_e as [[Hnull _] | [_ [Hcl _]]]; [ left; exact Hnull | right ].
  rewrite closures_satisfied in Hcl.
  destruct Hcl as (slot' & fdecls & vs & Hcis & Hrow).
  rewrite row_like_for_closures_lookup_iff in Hrow.
  destruct Hrow as (case & Hl & Hcase).
  destruct (single_tag_case known slot entry Hs) as (case0 & Hlk & Hent).
  rewrite Hlk in Hl. destruct (Pos.eqb_spec slot' slot) as [-> | Hne];
    [ | discriminate Hl ].
  injection Hl as <-. exists fdecls, vs, case0.
  split; [ exact Hcis | split; [ | split; [ exact Hent | exact Hcase ] ] ].
  rewrite Hlk, Pos.eqb_refl. reflexivity.
Qed.

(* A closure of the first run, rebuilt under a metadata that does not
   forget: a closure of the second run of the same set, at the same slot,
   with the value slots that the rebuild keeps, and allocations that hold
   one another, of which the one at the slot may be at a node where the
   metadata holds. Its projections are those of the first run, or
   closures. *)
Lemma closure_rebuilt_view (m : rewrite_metadata) (v v' : value)
    (slot : function_slot) (fdecls : list (function_slot * code_id))
    (vs : list (value_slot * value)) :
  forgets m = false -> rebuilt_under G sol m v v' ->
  closure_in_set statics v = Some (slot, fdecls, vs) ->
  ∃ (sites : function_slot -> node) (vs' : list (value_slot * value))
    (n : node),
    closure_in_set statics' v' = Some (slot, fdecls, vs')
    ∧ (∀ g : function_slot,
         is_bound g fdecls = true ->
         project_function_slot statics' slot g v'
         = project_function_slot statics slot g v
         ∨ ∃ c : value,
             project_function_slot statics slot g v = Some c
             ∧ ~ unallocated c)
    ∧ may_come_from sol (sites slot) n ∧ metadata_at sol m n
    ∧ set_sites G fdecls sites
    ∧ for_all_bindings
        (value_slot_rebuilt G sol (rebuilt_value G sol) fdecls sites vs') vs.
Proof.
  intros Hm [(c & -> & ->) | (n & Hv & Hat & Hhu)] Hcis;
    [ discriminate Hcis | ].
  specialize (Hhu Hm).
  destruct v as [c | code vs0 | slot0 fdecls0 vs0 | s | tag fields | elements
                | c | l | callee args arity];
    cbn [closure_in_set] in Hcis; try discriminate;
    cbn [rebuilt_value] in Hv; try contradiction.
  - destruct (lookup slot0 fdecls0) as [id0 | ] eqn:Hl0; cbn in Hcis;
      [ | discriminate ].
    injection Hcis as -> -> ->.
    destruct Hv as (sites & vs' & -> & Hmcf & Hsites & Hslots).
    exists sites, vs', n.
    split; [ cbn [closure_in_set]; rewrite Hl0; reflexivity | ].
    split; [ | split; [ exact Hmcf | split; [ exact Hat | ] ] ];
      [ | split; [ exact Hsites | exact Hslots ] ].
    intros g Hg. right. apply is_bound_lookup in Hg as (idg & Hg).
    exists (Value_closure_of_set g fdecls vs). split.
    + cbn [project_function_slot]. rewrite Pos.eqb_refl, Hl0, Hg.
      reflexivity.
    + intros [(c0 & Hc0) | (s0 & Hs0)]; discriminate.
  - destruct Hv as (-> & Hmcf & Hhus).
    destruct (lookup s (static_values statics)) as [allocated | ] eqn:Hls;
      [ | discriminate ].
    pose proof (Hst s allocated Hls (Hhus Hhu)) as Hsr.
    destruct allocated as [code0 vs0 | slot0 csyms fdecls0 vs0 | | | | | ];
      try discriminate.
    + cbn [static_rebuilt] in Hsr. contradiction.
    + destruct (lookup slot0 fdecls0) as [id0 | ] eqn:Hl0; cbn in Hcis;
        [ | discriminate ].
      injection Hcis as -> -> ->.
      cbn [static_rebuilt] in Hsr.
      destruct Hsr as (sites & vs' & Hls' & Hsite & Hsites & Hslots).
      exists sites, vs', n. split.
      * cbn [closure_in_set]. rewrite Hls', Hl0. reflexivity.
      * split.
        -- intros g _. left. cbn [project_function_slot].
           rewrite Hls, Hls'. reflexivity.
        -- rewrite Hsite. auto.
Qed.

(* [refine_sets_of_closures] keeps the set of an allocation of a closure
   that the metadata describes. *)
Lemma refine_keeps (m : rewrite_metadata) (slot : function_slot)
    (code_ids : list (function_slot * Or_unknown.t code_id))
    (mbl : must_be_local) (S0 : set_of_closures) (a : node) :
  forgets m = false -> describes sol m a ->
  g_constructor G a (Field_function_slot slot) a ->
  S0 ∈ sets -> closure_at S0 slot = a -> is_bound slot S0 = true ->
  (∀ p : function_slot * Or_unknown.t code_id,
     p ∈ code_ids -> is_bound (fst p) S0 = true) ->
  S0 ∈ fst (refine_sets_of_closures fx ctx m slot code_ids mbl).
Proof.
  intros Hm Hd Hself HS0 Hat Hbound Hcodes.
  pose proof (least_closed_tables is_local F T Hleast) as Hc'.
  assert (Hcand :
            S0 ∈ filter
                   (fun set =>
                      forallb (fun p => is_bound (fst p) set) code_ids)
                   (filter (fun set => is_bound slot set) sets)).
  { apply filter_In. split; [ apply filter_In; split; assumption | ].
    apply forallb_forall. intros p Hp. exact (Hcodes p Hp). }
  assert (Hhs : s_has_source sol a = true).
  { exact (has_source_function_slot _ _ _ Hc' a _ a Hself eq_refl). }
  unfold refine_sets_of_closures. cbv zeta.
  destruct m as [ | s | | us | ]; cbn [forgets describes] in Hm, Hd;
    try discriminate.
  - subst s. cbn [fst]. apply filter_In. split; [ exact Hcand | ].
    rewrite Hat. cbn [mem_node existsb]. rewrite node_eqb_refl.
    reflexivity.
  - cbn [fst]. apply filter_In. split; [ exact Hcand | ].
    rewrite Hat. unfold any_usage. cbn [rc_tables plain_context].
    change (s_any_usage sol a || false = true). rewrite Hd. reflexivity.
  - match goal with
    | |- context [if is_nil ?l then _ else _] =>
      destruct (is_nil l) eqn:Hnil
    end;
      cbn [fst]; apply filter_In; split; try exact Hcand; rewrite Hat.
    + destruct Hd as [Hau | Hus].
      * change (s_any_usage sol a
                || subset_nodes us (get_direct_usages ctx [a]) = true).
        rewrite Hau. reflexivity.
      * apply orb_true_iff. right. unfold subset_nodes.
        apply forallb_forall. intros u Hu. apply mem_node_In.
        apply get_direct_usages_iff. exists a.
        split; [ left; reflexivity | exact (Hus u Hu) ].
    + destruct (s_any_usage sol a) eqn:Hau.
      * apply orb_true_iff. right. rewrite Hfix_refine. exact Hau.
      * apply orb_true_iff. left. apply mem_node_In.
        destruct Hd as [Hau' | Hus];
          [ first [ discriminate Hau'
                  | exfalso; exact (bool_both _ Hau' Hau) ] | ].
        apply In_intersect_all.
        { intros Habs. rewrite Habs in Hnil. discriminate Hnil. }
        intros l Hl. apply in_flat_map in Hl as (u & Hu & Hl).
        rewrite get_direct_sources_one in Hl.
        destruct (any_source ctx u) eqn:Hasu; [ destruct Hl | ].
        destruct Hl as [<- | []].
        destruct (Hus u Hu) as [Hau_u _].
        destruct (usages_carry_sources is_local F T Hleast a u Hau_u)
          as [Hcarry Hany].
        assert (Hnas : s_any_source sol a = false).
        { apply not_true_iff_false. intros Has.
          change (s_any_source sol u = false) in Hasu.
          rewrite (Hany Has) in Hasu. discriminate Hasu. }
        pose proof (sources_constructor _ _ _ Hc' a _ a Hself Hnas) as Haa.
        destruct (Hcarry a Haa) as [Hua | Has_u];
          [ | exfalso; exact (bool_both _ Has_u Hasu) ].
        apply In_dedup_nodes, In_filter_has_source.
        split; [ apply sources_of_iff; exact Hua | exact Hhs ].
Qed.

(* What the rule of a closure is made of: the sets that
   [refine_sets_of_closures] keeps, the slots of each of them, which are
   joined, and those of the sources outside the unit. *)
Lemma closure_rule_inv (m : rewrite_metadata) (slot : function_slot)
    (am : Alloc_mode.t) (fts : finite_map (Or_unknown.t function_type))
    (cts vsts : finite_map flambda_type) (mn : bool)
    (spec : closure_rewrite) :
  closure_rule fx ctx m slot am (Closures_entry fts cts vsts) mn
  = Rule_closure spec ->
  ∃ (mbl0 mbl : must_be_local) (local_sets : list set_of_closures)
    (first : slots_metadata) (rest : list slots_metadata)
    (vsr : list (value_slot * rewrite_metadata))
    (fsr : list (function_slot * (rewrite_metadata * bool))),
    refine_sets_of_closures fx ctx m slot
      (map (fun g => (g, entry_code_id (Closures_entry fts cts vsts) g))
         (domain cts))
      mbl0
    = (local_sets, mbl)
    ∧ (∀ (set : set_of_closures) (e : slots_metadata),
         set ∈ local_sets ->
         slots_of_local_set ctx (domain vsts) (domain cts) set = Some e ->
         e ∈ first :: rest)
    ∧ List.fold_left (join_slots ctx) rest first = (vsr, fsr)
    ∧ spec
      = Closure_rewrite slot am
          (map
             (fun p : function_slot * (rewrite_metadata * bool) =>
                (fst p,
                 if snd (snd p) then Or_unknown.Unknown
                 else entry_code_id (Closures_entry fts cts vsts) (fst p)))
             fsr)
          (map
             (fun p : function_slot * (rewrite_metadata * bool) =>
                (fst p, fst (snd p)))
             fsr)
          vsr mn.
Proof.
  unfold closure_rule. cbv zeta.
  match goal with
  | |- (let '(_, _) := refine_sets_of_closures _ _ _ _ _ ?mbl0 in _) = _
       -> _ =>
    set (mbl_in := mbl0)
  end.
  destruct (refine_sets_of_closures fx ctx m slot _ mbl_in)
    as [local_sets mbl] eqn:Hr.
  match goal with
  | |- context [if existsb ?f local_sets then _ else _] =>
    destruct (existsb f local_sets); [ discriminate | ]
  end.
  match goal with
  | |- (match ?l with [] => _ | _ :: _ => _ end) = _ -> _ =>
    destruct l as [ | first rest] eqn:Hall; [ discriminate | ]
  end.
  destruct (List.fold_left (join_slots ctx) rest first) as [vsr fsr] eqn:Hf.
  intros [= <-].
  exists mbl_in, mbl, local_sets, first, rest, vsr, fsr.
  split; [ exact Hr | split; [ | split; [ exact Hf | reflexivity ] ] ].
  intros set e Hset He. rewrite <- Hall.
  assert (Hloc : e ∈ flat_map
                       (fun set =>
                          match
                            slots_of_local_set ctx (domain vsts) (domain cts)
                              set
                          with
                          | Some s => [s]
                          | None => []
                          end)
                       local_sets).
  { apply in_flat_map. exists set. rewrite He. split; [ exact Hset | ].
    left. reflexivity. }
  match goal with
  | |- e ∈ match ?x with None => _ | Some _ => _ end =>
    destruct x; [ right; exact Hloc | exact Hloc ]
  end.
Qed.

(* The rewriting of an alias holds of what the alias holds of, when it
   is a constant or a symbol. *)
Definition rw_alias (rw : rewrite_metadata -> flambda_type -> flambda_type)
    : Prop :=
  ∀ (m : rewrite_metadata) (t : flambda_type) (w : value),
    is_alias t = true -> type_in_fragment valuation t ->
    type_satisfied statics valuation w t ->
    type_satisfied statics' valuation' w (rw m t).

Lemma rewrite_arbitrary_alias (fuel : nat) :
  rw_sound (rewrite_arbitrary fx ctx tenv canon fuel) ->
  rw_alias (rewrite_arbitrary fx ctx tenv canon fuel).
Proof.
  intros Hrw m t w Hal Ht Hsat.
  unfold is_alias in Hal.
  destruct (get_alias_opt t) as [s | ] eqn:Ha; [ | discriminate ].
  pose proof (proj1 (get_alias_satisfied _ _ w t s Ha) Hsat) as [Hs Hkt].
  destruct (proj2 Ht s (kind t) w (get_alias_in_aliases t s Ha) Hs)
    as [(c & ->) | (sym & ->)].
  - apply (Hrw m t (Value_const c) (Value_const c) Ht);
      [ left; exists c; split; reflexivity | exact Hsat ].
  - destruct fuel as [ | fuel]; cbn [rewrite_arbitrary];
      [ exact (sat_unknown_of_kind _ _ t Hsat eq_refl) | ].
    rewrite Ha.
    destruct (Typing_env.get_canonical_simple_exn tenv
                (Some Name_mode.in_types) None s)
      as [[[x | sym' | c] | ] | ] eqn:Hcan;
      try exact (sat_unknown_of_kind _ _ t Hsat eq_refl).
    + destruct (fix_alias fx && forgets m);
        [ exact (sat_unknown_of_kind _ _ t Hsat eq_refl) | ].
      apply alias_satisfied. split; [ | exact Hkt ].
      change (value_of_name valuation' (canon (Name_var x) (kind t) m)
              = Some (Value_symbol sym)).
      rewrite (Hcopies s (Name_var x) (kind t) m Hcan). exact Hs.
    + destruct (fix_alias fx && forgets m);
        [ exact (sat_unknown_of_kind _ _ t Hsat eq_refl) | ].
      apply alias_satisfied. split; [ | exact Hkt ].
      change (value_of_name valuation' (canon (Name_symbol sym') (kind t) m)
              = Some (Value_symbol sym)).
      rewrite (Hcopies s (Name_symbol sym') (kind t) m Hcan). exact Hs.
    + exfalso. pose proof (canonical_eval s (Const c) Hcan) as Hceq.
      rewrite Hs in Hceq. discriminate Hceq.
Qed.

Lemma some_inj {A : Type} (x y : A) : Some x = Some y -> x = y.
Proof. intros H. injection H as H. exact H. Qed.

(* The rule of a closure, applied to a type that holds of a value of the
   first run, holds of what the second run has for it. *)
Lemma closure_sound (rw : rewrite_metadata -> flambda_type -> flambda_type)
    (m : rewrite_metadata) (t : flambda_type) (v v' : value)
    (spec : closure_rewrite) :
  rw_sound rw -> rw_alias rw ->
  type_in_fragment valuation t -> rebuilt_under G sol m v v' ->
  type_satisfied statics valuation v t ->
  rewrite_rule fx ctx tenv m t = Rule_closure spec ->
  type_satisfied statics' valuation' v' (rewrite_closure fx tenv rw spec t).
Proof.
  intros Hrw Hrwa Ht Hv Hsat Hrule.
  pose proof Hrule as Hrule0.
  unfold rewrite_rule in Hrule.
  destruct (forgets m) eqn:Hm; [ discriminate | ].
  destruct (kind t); try discriminate.
  destruct (Provers.meet_single_closures_entry tenv t)
    as [[[[[slot am] entry] ft] | | ] | ] eqn:Hmeet; try discriminate.
  destruct (meet_closures_head tenv t slot am entry ft Hmeet)
    as (e & known & isn & He & Hd & Hsingle).
  destruct (closure_of_type t e known am isn slot entry v He Hd Hsingle Hsat)
    as [-> | (fdecls & vs & case & Hcis & Hl & Hent & Hcase)].
  { rewrite (rebuilt_under_const G sol m Null v' Hv).
    apply (null_closure fx Hfix_null tenv statics' valuation' rw spec t).
    exact (null_closure_rule fx ctx tenv statics valuation Henv m t spec Hsat
             Hrule0). }
  destruct entry as [fts cts vsts].
  pose proof (expand_head_fragment valuation tenv Hfrag t e He Ht) as Hfr_e.
  unfold Expanded_type.to_type in Hfr_e. rewrite Hd in Hfr_e.
  destruct (closures_case_fragment valuation known am isn slot case fts cts
              vsts Hfr_e (lookup_in slot case known Hl) Hent)
    as (Hcl & Hvl & Hcta & Hvsf).
  destruct Hcase as (Hidx & Hentry & _).
  rewrite Hent in Hentry.
  apply closures_entry_lookup_iff in Hentry as (Hfts & Hcts_sat & Hvsts_sat).
  destruct (index_slots fdecls vs (case_index case) Hidx)
    as [Hidx_f Hidx_v].
  destruct (closure_rebuilt_view m v v' slot fdecls vs Hm Hv Hcis)
    as (sites & vs' & n & Hcis' & Hproj & Hmcf & Hat & Hsites & Hslots).
  pose proof (closure_in_set_slot statics v slot fdecls vs Hcis) as Hslot_b.
  destruct (proj1 (is_bound_lookup slot fdecls) Hslot_b) as [id Hlslot].
  destruct (Hsets fdecls sites slot id Hlslot Hsites)
    as (S0 & HS0 & HS0in & HS0b).
  (* The allocation of the closure, which the metadata describes. *)
  assert (Hself : g_constructor G (sites slot) (Field_function_slot slot)
                    (sites slot)).
  { apply (proj2 (Hsites slot id Hlslot slot (sites slot))).
    split; [ exact Hslot_b | reflexivity ]. }
  assert (Hhs : s_has_source sol (sites slot) = true).
  { exact (has_source_function_slot _ _ _ Hc _ _ _ Hself eq_refl). }
  assert (Hdesc : describes sol m (sites slot)).
  { exact (Hat (sites slot) Hmcf Hhs). }
  (* Its set. *)
  set (nodes0 := map (fun p : function_slot * (node * Or_unknown.t code_id) =>
                        (fst p, fst (snd p))) S0).
  assert (Hnodes : ∀ (g : function_slot) (c : node),
             (g, c) ∈ nodes0 -> is_bound g fdecls = true ∧ c = sites g).
  { intros g c Hin. apply in_map_iff in Hin as ([g' [c' code]] & Hp & Hin).
    cbn [fst snd] in Hp. injection Hp as <- <-.
    exact (HS0in g' c' code Hin). }
  assert (HS0slot : is_bound slot S0 = true) by exact (HS0b slot Hslot_b).
  assert (Hat_slot : closure_at S0 slot = sites slot).
  { unfold closure_at.
    destruct (proj1 (is_bound_lookup slot S0) HS0slot) as [[c code] HlS0].
    rewrite HlS0.
    exact (proj2 (HS0in slot c code (lookup_in slot (c, code) S0 HlS0))). }
  assert (Hcodes :
            ∀ p : function_slot * Or_unknown.t code_id,
              p ∈ map (fun g => (g, entry_code_id (Closures_entry fts cts vsts)
                                      g))
                    (domain cts) ->
              is_bound (fst p) S0 = true).
  { intros p Hp. apply in_map_iff in Hp as (g & <- & Hg). cbn [fst].
    apply HS0b. rewrite <- among_domain. apply Hidx_f.
    apply (proj1 (all_among _ _) Hcl). apply among_In. exact Hg. }
  destruct (closure_rule_inv m slot am fts cts vsts _ spec Hrule)
    as (mbl0 & mbl & local_sets & first & rest & vsr & fsr
        & Hr & Hmem & Hf & Hspec).
  pose proof (refine_keeps m slot _ mbl0 S0 (sites slot) Hm Hdesc Hself HS0
                Hat_slot HS0slot Hcodes) as HS0loc.
  rewrite Hr in HS0loc. cbn [fst] in HS0loc.
  (* The value slots of the closure that a closure of the set uses. *)
  assert (Hheld :
            ∀ (w : value_slot) (x : value),
              lookup w vs = Some x ->
              (∃ (g : function_slot) (c : node),
                 (g, c) ∈ nodes0
                 ∧ s_field_used sol c (Field_value_slot w) = true) ->
              ∃ (from : node) (x' : value),
                slot_held G fdecls sites w from
                ∧ s_has_source sol from = true ∧ s_has_usage sol from = true
                ∧ lookup w vs' = Some x' ∧ rebuilt_value G sol x from x').
  { intros w x Hx (g & c & Hin & Hfu).
    destruct (Hnodes g c Hin) as [Hgb ->].
    destruct (proj1 (is_bound_lookup g fdecls) Hgb) as [idg Hg].
    exact (proj1 (for_all_bindings_lookup _ _ vs) Hslots w x Hx g idg Hg
             Hfu). }
  assert (Hnodes_held :
            ∀ (w : value_slot) (from : node),
              slot_held G fdecls sites w from ->
              ∀ (g : function_slot) (c : node),
                (g, c) ∈ nodes0 ->
                ∀ b : node,
                  g_constructor G c (Field_value_slot w) b <-> b = from).
  { intros w from Hsh g c Hin b. destruct (Hnodes g c Hin) as [Hgb ->].
    destruct (proj1 (is_bound_lookup g fdecls) Hgb) as [idg Hg].
    exact (Hsh g idg Hg b). }
  assert (Hin_vs : ∀ w : value_slot,
             w ∈ domain vsts -> ∃ x : value, lookup w vs = Some x).
  { intros w Hw. apply is_bound_lookup. rewrite <- among_domain.
    apply Hidx_v. apply (proj1 (all_among _ _) Hvl). apply among_In.
    exact Hw. }
  (* The slots of the set. *)
  destruct (slots_of_local_set_shape ctx (domain vsts) (domain cts) S0)
    as (fsl & Hshape & Hfsl).
  destruct (local_value_slots_some ctx nodes0 (domain vsts)) as (vsl & Hvsl).
  { intros w Hw. destruct (Hin_vs w Hw) as [x Hx].
    destruct (follow_field_for_set_cases ctx nodes0 w) as [Hal | Huse];
      [ rewrite Hal; discriminate | ].
    destruct (Hheld w x Hx Huse) as (from & x' & Hsh & Hhsf & _).
    exact (follow_field_for_set_some is_local F T Hleast sets nodes0 w from
             (Hnodes_held w from Hsh) Hhsf). }
  assert (He0 : slots_of_local_set ctx (domain vsts) (domain cts) S0
                = Some (vsl, fsl)).
  { rewrite Hshape. fold nodes0. rewrite Hvsl. reflexivity. }
  pose proof (Hmem S0 (vsl, fsl) HS0loc He0) as He0in.
  pose proof (fold_join_covered ctx rest first (vsl, fsl) He0in) as Hcov.
  rewrite Hf in Hcov. destruct Hcov as [Hcov_v Hcov_f].
  cbn [fst snd] in Hcov_v, Hcov_f.
  (* The value slots of the rewritten type. *)
  assert (Hvs_kept :
            ∀ (w : value_slot) (mw : rewrite_metadata),
              lookup w vsr = Some mw ->
              ∃ (ty : flambda_type) (x : value) (from : node) (x' : value),
                lookup w vsts = Some ty ∧ lookup w vs = Some x
                ∧ lookup w vs' = Some x' ∧ rebuilt_value G sol x from x'
                ∧ s_has_usage sol from = true ∧ metadata_at sol mw from).
  { intros w mw Hw. destruct (Hcov_v w mw Hw) as (m0 & Hm0 & Hdm).
    destruct (local_value_slots_spec ctx nodes0 (domain vsts) vsl Hvsl w m0
                Hm0) as (Hwin & Hfol & Hne).
    destruct (Hin_vs w Hwin) as [x Hx].
    assert (Hwt : is_bound w vsts = true).
    { rewrite <- among_domain. apply among_In. exact Hwin. }
    destruct (proj1 (is_bound_lookup w vsts) Hwt) as [ty Hty].
    destruct (follow_field_for_set_cases ctx nodes0 w) as [Hal | Huse].
    { rewrite Hal in Hfol. injection Hfol as <-. contradiction. }
    destruct (Hheld w x Hx Huse)
      as (from & x' & Hsh & _ & Hhuf & Hx' & Hrv).
    exists ty, x, from, x'.
    split; [ exact Hty | split; [ exact Hx | split; [ exact Hx' | ] ] ].
    split; [ exact Hrv | split; [ exact Hhuf | ] ].
    intros a Ha Hha. apply Hdm.
    exact (follow_field_for_set_at is_local F T Hleast sets nodes0 w from m0
             (Hnodes_held w from Hsh) Hfol a Ha Hha). }
  (* The function slots of the rewritten type. *)
  assert (Hfs_kept :
            ∀ (g : function_slot) (p : rewrite_metadata * bool),
              lookup g fsr = Some p ->
              is_bound g cts = true ∧ is_bound g fdecls = true).
  { intros g p Hg. pose proof (Hcov_f g p Hg) as Hgb.
    destruct (proj1 (is_bound_lookup g fsl) Hgb) as [p0 Hp0].
    apply lookup_in in Hp0.
    assert (Hgin : g ∈ domain cts).
    { rewrite <- Hfsl. apply in_map_iff. exists (g, p0). auto. }
    assert (Hgc : among g (domain cts) = true) by (apply among_In; exact Hgin).
    split; [ rewrite <- among_domain; exact Hgc | ].
    rewrite <- among_domain. apply Hidx_f.
    exact (proj1 (all_among _ _) Hcl g Hgc). }
  (* The type. *)
  rewrite Hspec. unfold rewrite_closure. cbv zeta.
  cbn [cr_slot cr_function_types cr_closure_types cr_value_slots
    cr_alloc_mode cr_maybe_null].
  rewrite He.
  match goal with
  | |- context [at_least_this_closure ?s ?a ?b ?c ?d] =>
    assert (Hty' : type_satisfied statics' valuation' v'
                     (at_least_this_closure s a b c d))
  end.
  { apply at_least_this_closure_satisfied. exists fdecls, vs'.
    split; [ exact Hcis' | split; [ | split ] ].
    - rewrite !domain_map_keys. apply all_among. intros g Hg.
      rewrite among_domain in Hg |- *.
      destruct (proj1 (is_bound_lookup g fsr) Hg) as [p Hp].
      exact (proj2 (Hfs_kept g p Hp)).
    - rewrite domain_map_keys. apply all_among. intros w Hw.
      rewrite among_domain in Hw |- *.
      destruct (proj1 (is_bound_lookup w vsr) Hw) as [mw Hmw].
      destruct (Hvs_kept w mw Hmw) as (ty & x & from & x' & _ & _ & Hx' & _).
      apply is_bound_lookup. exists x'. exact Hx'.
    - apply closures_entry_lookup_iff. split; [ | split ].
      + intros g ft' code Hg Hcode.
        rewrite !lookup_map_keys in Hg.
        destruct (lookup g fsr) as [[mg erase] | ] eqn:Hgf;
          [ | discriminate Hg ].
        cbn [fst snd] in Hg. destruct erase; [ discriminate Hg | ].
        unfold entry_code_id in Hg.
        cbn [closures_entry_find_function_type] in Hg.
        destruct (lookup g fts) as [[ft1 | ] | ] eqn:Hlf; try discriminate Hg.
        injection Hg as <-. cbn [function_type_code_id].
        exact (Hfts g ft1 code Hlf Hcode).
      + intros g ty' closure' Hg Hproj'.
        rewrite !lookup_map_keys in Hg.
        destruct (lookup g fsr) as [[mg erase] | ] eqn:Hgf;
          [ | discriminate Hg ].
        apply some_inj in Hg. cbn [fst snd] in Hg. subst ty'.
        destruct (Hfs_kept g _ Hgf) as [Hgc Hgd].
        rewrite (destructure_closure_type e known am isn slot fts cts vsts g
                   Hd Hsingle).
        destruct (proj1 (is_bound_lookup g cts) Hgc) as [tyg Htyg].
        rewrite Htyg.
        destruct (Hcta g tyg (lookup_in g tyg cts Htyg)) as [Hal Hfrg].
        destruct (Hproj g Hgd) as [Heq | (c & Hpc & Hnu)].
        * rewrite Heq in Hproj'.
          exact (Hrwa mg tyg closure' Hal Hfrg
                   (Hcts_sat g tyg closure' Htyg Hproj')).
        * exfalso. pose proof (Hcts_sat g tyg c Htyg Hpc) as Hsc.
          unfold is_alias in Hal.
          destruct (get_alias_opt tyg) as [s | ] eqn:Ha; [ | discriminate ].
          destruct (proj1 (get_alias_satisfied _ _ c tyg s Ha) Hsc)
            as [Hs _].
          exact (Hnu (proj2 Hfrg s (kind tyg) c (get_alias_in_aliases tyg s Ha)
                        Hs)).
      + intros w ty' contents Hw Hcont.
        rewrite lookup_map_keys in Hw.
        destruct (lookup w vsr) as [mw | ] eqn:Hwv; [ | discriminate Hw ].
        apply some_inj in Hw. cbn [fst snd] in Hw. subst ty'.
        destruct (Hvs_kept w mw Hwv)
          as (ty & x & from & x' & Hty & Hx & Hx' & Hrv & Hhuf & Hatf).
        rewrite Hx' in Hcont. injection Hcont as <-.
        rewrite (destructure_env_var e known am isn slot fts cts vsts w Hd
                   Hsingle), Hty.
        apply (Hrw mw ty x x').
        * exact (Hvsf w ty (lookup_in w ty vsts Hty)).
        * right. exists from.
          split; [ exact Hrv | split; [ exact Hatf | intros _; exact Hhuf ] ].
        * exact (Hvsts_sat w ty x Hty Hx). }
  destruct (fix_null fx && expanded_maybe_null tenv t); [ | exact Hty' ].
  unfold at_least_this_closure, create_closures, non_null_value in Hty' |- *.
  apply maybe_null_of_not_null. exact Hty'.
Qed.

(* The rewriting, by induction on its fuel. *)
Lemma rewrite_sound (fuel : nat) :
  rw_sound (rewrite_arbitrary fx ctx tenv canon fuel)
  ∧ rw_sound (rewrite fx ctx tenv canon fuel).
Proof.
  induction fuel as [ | fuel [IHa IHr]]; split; intros m t v v' Ht Hv Hsat.
  - exact (sat_unknown_of_kind v v' t Hsat (rebuilt_under_kind _ _ m v v' Hv)).
  - exact (sat_unknown_of_kind v v' t Hsat (rebuilt_under_kind _ _ m v v' Hv)).
  - cbn [rewrite_arbitrary].
    destruct (get_alias_opt t) as [s | ] eqn:Ha;
      [ | exact (IHr m t v v' Ht Hv Hsat) ].
    pose proof (proj1 (get_alias_satisfied _ _ v t s Ha) Hsat) as [Hs Hkt].
    pose proof (proj2 Ht s (kind t) v (get_alias_in_aliases t s Ha) Hs)
      as Hunalloc.
    destruct (Typing_env.get_canonical_simple_exn tenv
                (Some Name_mode.in_types) None s)
      as [[[x | sym | c] | ] | ] eqn:Hcan;
      try exact (sat_unknown_of_kind v v' t Hsat
                   (rebuilt_under_kind _ _ m v v' Hv)).
    + destruct (fix_alias fx && forgets m);
        [ exact (sat_unknown_of_kind v v' t Hsat
                   (rebuilt_under_kind _ _ m v v' Hv)) | ].
      rewrite (rebuilt_under_unallocated _ _ m v v' Hunalloc Hv).
      apply alias_satisfied. split; [ | exact Hkt ].
      change (value_of_name valuation' (canon (Name_var x) (kind t) m)
              = Some v).
      rewrite (Hcopies s (Name_var x) (kind t) m Hcan). exact Hs.
    + destruct (fix_alias fx && forgets m);
        [ exact (sat_unknown_of_kind v v' t Hsat
                   (rebuilt_under_kind _ _ m v v' Hv)) | ].
      rewrite (rebuilt_under_unallocated _ _ m v v' Hunalloc Hv).
      apply alias_satisfied. split; [ | exact Hkt ].
      change (value_of_name valuation' (canon (Name_symbol sym) (kind t) m)
              = Some v).
      rewrite (Hcopies s (Name_symbol sym) (kind t) m Hcan). exact Hs.
    + pose proof (canonical_eval s (Const c) Hcan) as Hceq.
      rewrite Hs in Hceq. cbn in Hceq. injection Hceq as <-.
      apply (IHr m (type_for_const c) (Value_const c) v').
      * apply type_for_const_fragment.
      * exact Hv.
      * apply type_for_const_satisfied.
  - cbn [rewrite].
    destruct (rewrite_rule fx ctx tenv m t) as [ | | spec] eqn:Hrule.
    + exact (identity_sound (rewrite_arbitrary fx ctx tenv canon fuel) IHa
               (proj1 (rewrite_forgets fuel)) m t v v'
               (rule_identity_not_forgets m t Hrule) Ht Hv Hsat).
    + exact (sat_unknown_of_kind v v' t Hsat
               (rebuilt_under_kind _ _ m v v' Hv)).
    + exact (closure_sound (rewrite_arbitrary fx ctx tenv canon fuel) m t v
               v' spec IHa (rewrite_arbitrary_alias fuel IHa) Ht Hv Hsat
               Hrule).
Qed.

End Sound.

Theorem rewrite_type_sound : rewrite_type_sound_statement.
Proof.
  intros is_local F T sets tenv canon statics statics' valuation valuation'
    fuel m t v v' Hleast Hlocal Hsets Henv Hfrag Halloc Hst Hcopies Ht Hv
    Hsat.
  exact (proj1 (rewrite_sound all_fixes eq_refl eq_refl eq_refl eq_refl
                  is_local F T sets tenv canon statics statics' valuation
                  valuation' (least_closed_tables is_local F T Hleast) Hleast
                  Hlocal Hsets Henv Hfrag Halloc Hst Hcopies fuel)
           m t v v' Ht Hv Hsat).
Qed.

Theorem rewrite_symbol_type_sound : rewrite_symbol_type_sound_statement.
Proof.
  intros is_local F T sets tenv canon statics statics' valuation valuation' s
    allocated Hleast Hlocal Hsets Henv Hfrag Halloc Hst Hcopies Hl.
  unfold rewrite_symbol_type, rewrite_concrete_type_of.
  set (ty := concrete_type_of tenv (Name_symbol s) Value).
  assert (Hty : type_in_fragment valuation ty
                ∧ type_satisfied statics valuation (Value_symbol s) ty).
  { unfold ty, concrete_type_of.
    destruct (Typing_env.find tenv (Name_symbol s) (Some Value))
      as [ty0 | ] eqn:Hf.
    - split; [ exact (find_fragment valuation tenv Hfrag _ _ ty0 Hf) | ].
      exact (find_satisfied_strictly statics valuation tenv (Name_symbol s)
               (Some Value) ty0 (Value_symbol s) Henv Hf eq_refl).
    - split; [ apply unknown_fragment | apply unknown_satisfied;
                                         reflexivity ]. }
  destruct Hty as [Hfr Hsat].
  pose proof (metadata_for_name_at (plain_context is_local F T sets)
                (Node_symbol s)) as [Hat Hhu].
  apply (proj2 (rewrite_sound all_fixes eq_refl eq_refl eq_refl eq_refl
                  is_local F T sets tenv canon statics statics' valuation
                  valuation' (least_closed_tables is_local F T Hleast) Hleast
                  Hlocal Hsets Henv Hfrag Halloc Hst Hcopies
                  (rewrite_fuel tenv ty))
           _ ty (Value_symbol s) (Value_symbol s) Hfr);
    [ | exact Hsat ].
  right. exists (Node_symbol s).
  split; [ | split; [ exact Hat | exact Hhu ] ].
  cbn [rebuilt_value].
  split; [ reflexivity | split; [ exact (Halloc s allocated Hl) | auto ] ].
Qed.

(* ======================================================================== *)
(* The result types of code                                                 *)
(* ======================================================================== *)

Lemma kind_bottom (k : flambda_kind) : kind (bottom k) = k.
Proof. destruct k as [ | [] | | ]; reflexivity. Qed.

(* The type of a name found at no kind is found at its own. *)
Lemma find_with_kind (tenv : Typing_env.t) (n : name) (ty : flambda_type) :
  Typing_env.find tenv n None = Some ty ->
  Typing_env.find tenv n (Some (kind ty)) = Some ty.
Proof.
  unfold Typing_env.find, Typing_env.find_with_binding_time_and_mode'.
  destruct (Keyed.lookup name_eqb n (Typing_env.names_to_types tenv))
    as [[ty0 bt] | ] eqn:Hl.
  - cbn [Typing_env.check_optional_kind_matches].
    destruct (Typing_env.is_bottom tenv); cbn; intros [= <-];
      [ rewrite kind_bottom | ]; rewrite kind_eqb_refl; reflexivity.
  - destruct n as [x | sym]; [ discriminate | ].
    cbn [Typing_env.check_optional_kind_matches andb].
    destruct (among sym (Typing_env.defined_symbols tenv)); [ | discriminate ].
    cbn. intros [= <-]. reflexivity.
Qed.

Theorem rewrite_result_type_sound : rewrite_result_type_sound_statement.
Proof.
  intros is_local F T sets tenv canon statics statics' valuation valuation'
    code_var type_var v v' Hleast Hlocal Hsets Henv Hfrag Halloc Hst Hcopies
    Hfind Hl Hv.
  unfold rewrite_result_type, rewrite_concrete_type_of, kind_of_name.
  destruct (Typing_env.find tenv (Name_var type_var) None) as [ty0 | ] eqn:Hf;
    [ | contradiction ].
  unfold concrete_type_of. rewrite (find_with_kind tenv _ ty0 Hf).
  pose proof (find_fragment valuation tenv Hfrag _ _ ty0 Hf) as Hfr.
  pose proof (find_satisfied_strictly statics valuation tenv (Name_var type_var)
                None ty0 v Henv Hf Hl) as Hsat.
  pose proof (metadata_for_name_at (plain_context is_local F T sets)
                (Node_variable code_var)) as [Hat Hhu].
  apply (proj2 (rewrite_sound all_fixes eq_refl eq_refl eq_refl eq_refl
                  is_local F T sets tenv canon statics statics' valuation
                  valuation' (least_closed_tables is_local F T Hleast) Hleast
                  Hlocal Hsets Henv Hfrag Halloc Hst Hcopies
                  (rewrite_fuel tenv ty0))
           _ ty0 v v' Hfr);
    [ | exact Hsat ].
  right. exists (Node_variable code_var).
  split; [ exact Hv | split; [ exact Hat | exact Hhu ] ].
Qed.

(* ======================================================================== *)
(* The two runs                                                             *)
(* ======================================================================== *)

Theorem exported_types_sound : exported_types_sound_statement.
Proof.
  intros flag is_local u e T preserve_direct_calls sets tenv canon valuation
    valuation' n s k vs hp rd sym allocated Hlocal Hleast Hsets Hrel Hrun
    Hstep Henv Hfrag Hcopies Hl.
  destruct (Hrel n s k vs hp rd Hrun Hstep)
    as [Halloc (m & t & vs' & rd' & Hrun' & Hstep' & Hst)].
  exists m, t, vs', rd'. split; [ exact Hrun' | split; [ exact Hstep' | ] ].
  exact (rewrite_symbol_type_sound is_local (traverse_unit flag u e) T sets
           tenv canon (state_statics s) (state_statics t) valuation
           valuation' sym allocated
           Hleast Hlocal Hsets Henv Hfrag Halloc Hst Hcopies Hl).
Qed.

(* ======================================================================== *)
(* The fixes that are needed                                                *)
(* ======================================================================== *)

Lemma may_come_from_self (sol : solution) (a : node) :
  s_has_source sol a = true ->
  s_any_source sol a = true ∨ s_sources sol a a = true ->
  may_come_from sol a a.
Proof.
  intros Hhs Hsrc. split; [ exact Hhs | split; [ exact Hsrc | ] ].
  split; [ auto | split; [ auto | ] ]. intros _ u Hu. left. exact Hu.
Qed.

(* The empty environment, with the valuation that satisfies it. *)
Lemma create_valuation (statics : static_env) :
  ∃ valuation : env,
    typing_env_satisfied_strictly statics valuation Typing_env.create
    ∧ env_in_fragment valuation Typing_env.create
    ∧ copies_valued Typing_env.create (fun n _ _ => n) valuation valuation.
Proof.
  destruct (create_satisfied statics []) as [_ (valuation & Hsat & _)].
  exists valuation. split; [ exact Hsat | split ].
  - intros n ty bt H. vm_compute in H. discriminate H.
  - intros s x k m Hcan. unfold value_of_name.
    exact (get_canonical_simple_satisfied _ _ _ _ _ _ _ _ Hsat Hcan).
Qed.

(* A graph with no function slot has no set of closures. *)
Lemma sets_of_graph_no_closures (F : facts) :
  (∀ (a : node) (g : function_slot) (b : node),
     (a, Field_function_slot g, b) ∈ f_constructor F -> False) ->
  sets_of_graph (graph_of_facts F) [].
Proof.
  intros Hno fd sites slot id Hslot Hsites. exfalso.
  apply (Hno (sites slot) slot (sites slot)).
  apply (proj2 (Hsites slot id Hslot slot (sites slot))).
  split; [ unfold is_bound; rewrite Hslot; reflexivity | reflexivity ].
Qed.

Lemma boxed_counterexample (fx : fixes) :
  fix_boxed fx = false -> ~ rewrite_type_sound_with fx.
Proof.
  intros Hfx H. unfold rewrite_type_sound_with in H.
  destruct (create_valuation empty_static_env)
    as (valuation & Henv & Hfrag & Hcopies).
  pose proof (H BoxedCounterexample.is_local BoxedCounterexample.graph
                BoxedCounterexample.solution [] Typing_env.create
                (fun n _ _ => n) empty_static_env empty_static_env valuation
                valuation 5 BoxedCounterexample.metadata
                BoxedCounterexample.boxed_type BoxedCounterexample.boxed
                BoxedCounterexample.rebuilt) as Hsound.
  clear H.
  apply type_satisfied_b_iff in Hsound.
  - destruct fx as [b1 b2 b3 b4]. cbn in Hfx. subst b1.
    vm_compute in Hsound. discriminate Hsound.
  - apply (check_least_sound _ _ 100). vm_compute. reflexivity.
  - intros f Hf. discriminate Hf.
  - apply sets_of_graph_no_closures. intros a g b Hin. cbn in Hin.
    repeat (destruct Hin as [Hin | Hin]; [ discriminate Hin | ]).
    exact Hin.
  - exact Henv.
  - exact Hfrag.
  - intros s allocated Hl. discriminate Hl.
  - intros s allocated Hl. discriminate Hl.
  - exact Hcopies.
  - split; [ vm_compute; reflexivity | ].
    intros s k w Hin. vm_compute in Hin. destruct Hin.
  - right. exists BoxedCounterexample.x10. split; [ | split ].
    + cbn [rebuilt_value].
      exists BoxedCounterexample.x10, Boxable_naked_float, Node_all_constants.
      split; [ apply may_come_from_self; vm_compute; auto | ].
      split; [ vm_compute; reflexivity | ].
      split; [ reflexivity | ].
      split; [ cbn; left; reflexivity | vm_compute; reflexivity ].
    + intros a (_ & [Has | Hsrc] & _) _;
        [ vm_compute in Has; discriminate Has | ].
      cbn [describes]. cbn [s_sources solution_of_tables] in Hsrc.
      apply mem_pair_In in Hsrc. cbn in Hsrc.
      destruct Hsrc as [[= <-] | []]. reflexivity.
    + intros _. vm_compute. reflexivity.
  - apply type_satisfied_b_iff. vm_compute. reflexivity.
Qed.

Theorem boxed_fix_needed : boxed_fix_needed_statement.
Proof. apply boxed_counterexample. reflexivity. Qed.

Theorem compiler_rewrite_unsound : compiler_rewrite_unsound_statement.
Proof. apply boxed_counterexample. reflexivity. Qed.

Theorem boxed_result_fix_needed : boxed_result_fix_needed_statement.
Proof.
  intros H. unfold rewrite_result_type_sound_with in H.
  set (valuation := BoxedResultCounterexample.valuation).
  set (tenv := BoxedResultCounterexample.tenv).
  assert (Hlookup : ∀ (n : name) (ty : flambda_type)
                      (bt : Binding_time.With_name_mode.t),
             Keyed.lookup name_eqb n (Typing_env.names_to_types tenv)
             = Some (ty, bt) ->
             n = Name_var 20%positive
             ∧ (ty = BoxedCounterexample.boxed_type ∨ ty = unknown Value)).
  { intros n ty bt Hl.
    apply (keyed_lookup_in name_eqb name_eqb_eq) in Hl.
    vm_compute in Hl.
    destruct Hl as [[= <- <- _] | [[= <- <- _] | []]];
      split; auto. }
  assert (Henv : typing_env_satisfied_strictly empty_static_env valuation tenv).
  { split; [ vm_compute; reflexivity | split ].
    - intros n ty bt w Hl Hw. right.
      destruct (Hlookup n ty bt Hl) as [-> [-> | ->]];
        vm_compute in Hw; injection Hw as <-;
        apply type_satisfied_b_iff; vm_compute; reflexivity.
    - change (Typing_env.aliases tenv) with Aliases.empty.
      apply empty_respected. }
  pose proof (H BoxedCounterexample.is_local BoxedCounterexample.graph
                BoxedCounterexample.solution [] tenv (fun n _ _ => n)
                empty_static_env empty_static_env valuation valuation
                10%positive 20%positive BoxedCounterexample.boxed
                BoxedCounterexample.rebuilt) as Hsound.
  clear H.
  apply type_satisfied_b_iff in Hsound.
  - vm_compute in Hsound. discriminate Hsound.
  - apply (check_least_sound _ _ 100). vm_compute. reflexivity.
  - intros f Hf. discriminate Hf.
  - apply sets_of_graph_no_closures. intros a g b Hin. cbn in Hin.
    repeat (destruct Hin as [Hin | Hin]; [ discriminate Hin | ]).
    exact Hin.
  - exact Henv.
  - intros n ty bt Hl. destruct (Hlookup n ty bt Hl) as [_ [-> | ->]];
      [ | apply unknown_fragment ].
    split; [ vm_compute; reflexivity | ].
    intros s k w Hin. vm_compute in Hin. destruct Hin.
  - intros s allocated Hl. discriminate Hl.
  - intros s allocated Hl. discriminate Hl.
  - intros s x k m Hcan. unfold value_of_name.
    exact (get_canonical_simple_satisfied _ _ _ _ _ _ _ _ Henv Hcan).
  - vm_compute. discriminate.
  - reflexivity.
  - cbn [rebuilt_value].
    exists BoxedCounterexample.x10, Boxable_naked_float, Node_all_constants.
    split; [ apply may_come_from_self; vm_compute; auto | ].
    split; [ vm_compute; reflexivity | ].
    split; [ reflexivity | ].
    split; [ cbn; left; reflexivity | vm_compute; reflexivity ].
Qed.

Theorem alias_fix_needed : alias_fix_needed_statement.
Proof.
  intros H. unfold rewrite_type_sound_with in H.
  destruct (create_valuation empty_static_env)
    as (valuation & Henv & Hfrag & Hcopies).
  pose proof (H AliasCounterexample.is_local AliasCounterexample.graph
                AliasCounterexample.solution [] Typing_env.create
                (fun n _ _ => n) empty_static_env empty_static_env valuation
                valuation 5 AliasCounterexample.metadata
                AliasCounterexample.block_type AliasCounterexample.block
                AliasCounterexample.rebuilt) as Hsound.
  clear H.
  apply type_satisfied_b_iff in Hsound.
  - vm_compute in Hsound. discriminate Hsound.
  - apply (check_least_sound _ _ 100). vm_compute. reflexivity.
  - intros f Hf. discriminate Hf.
  - apply sets_of_graph_no_closures. intros a g b Hin. cbn in Hin.
    repeat (destruct Hin as [Hin | Hin]; [ discriminate Hin | ]).
    exact Hin.
  - exact Henv.
  - exact Hfrag.
  - intros s allocated Hl. discriminate Hl.
  - intros s allocated Hl. discriminate Hl.
  - exact Hcopies.
  - split; [ vm_compute; reflexivity | ].
    intros s k w Hin Hev. vm_compute in Hin.
    destruct Hin as [[= <- _] | [[= <- _] | []]];
      cbn in Hev; injection Hev as <-; right; eexists; reflexivity.
  - right. exists AliasCounterexample.x10. split; [ | split ].
    + cbn [rebuilt_value].
      exists AliasCounterexample.x10,
        [Value_symbol 50%positive; Value_const (Poison Value)].
      split; [ apply may_come_from_self; vm_compute; auto | ].
      split; [ vm_compute; reflexivity | ].
      split; [ reflexivity | ].
      cbn [fields_rebuilt kind_of_value].
      split; [ | split; [ | exact I ] ].
      * exists AliasCounterexample.s50.
        split; [ cbn; left; reflexivity | split ].
        -- intros _. split; [ vm_compute; reflexivity | ].
           cbn [rebuilt_value]. split; [ reflexivity | split; [ | auto ] ].
           apply may_come_from_self; vm_compute; auto.
        -- intros Hfu. vm_compute in Hfu. discriminate Hfu.
      * exists AliasCounterexample.s50.
        split; [ cbn; right; left; reflexivity | split ].
        -- intros Hfu. vm_compute in Hfu. discriminate Hfu.
        -- intros _. reflexivity.
    + intros a (_ & [Has | Hsrc] & _) _;
        [ vm_compute in Has; discriminate Has | ].
      cbn [s_sources solution_of_tables] in Hsrc.
      apply mem_pair_In in Hsrc. cbn in Hsrc.
      destruct Hsrc as [[= <-] | [[= Habs] | []]].
      cbn [describes]. right. intros u [<- | []].
      split; vm_compute; reflexivity.
    + intros _. vm_compute. reflexivity.
  - apply type_satisfied_b_iff. vm_compute. reflexivity.
Qed.

(* The function slots of the closures of the unit of the refinement are
   those of its two sets. *)
Lemma refine_graph_function_slots (a : node) (g : function_slot) (b : node) :
  g_constructor (graph_of_facts RefineCounterexample.graph) a
    (Field_function_slot g) b ->
  g = 70%positive
  ∧ ((a = RefineCounterexample.x10 ∧ b = RefineCounterexample.x10)
     ∨ (a = RefineCounterexample.x11 ∧ b = RefineCounterexample.x11)).
Proof.
  cbn [graph_of_facts g_constructor RefineCounterexample.graph f_constructor].
  intros Hin.
  destruct Hin as [H | [H | [H | [H | []]]]]; injection H as Ha Hg Hb;
    try discriminate Hg; subst; auto.
Qed.

Theorem refine_fix_needed : refine_fix_needed_statement.
Proof.
  intros H. unfold rewrite_type_sound_with in H.
  destruct (create_valuation empty_static_env)
    as (valuation & Henv & Hfrag & Hcopies).
  pose proof (H RefineCounterexample.is_local RefineCounterexample.graph
                RefineCounterexample.solution RefineCounterexample.sets
                Typing_env.create (fun n _ _ => n) empty_static_env
                empty_static_env valuation valuation 5
                RefineCounterexample.metadata
                RefineCounterexample.closure_type RefineCounterexample.closure
                RefineCounterexample.rebuilt) as Hsound.
  clear H.
  apply type_satisfied_b_iff in Hsound.
  - vm_compute in Hsound. discriminate Hsound.
  - apply (check_least_sound _ _ 100). vm_compute. reflexivity.
  - intros f Hf. destruct f; try discriminate Hf. left. eexists. reflexivity.
  - intros fd sites slot id Hslot Hsites.
    assert (Hb : is_bound slot fd = true).
    { unfold is_bound. rewrite Hslot. reflexivity. }
    assert (Hdom : ∀ g : function_slot,
               is_bound g fd = true ->
               g = 70%positive ∧ sites g = sites slot).
    { intros g Hg.
      destruct (refine_graph_function_slots (sites slot) g (sites g)
                  (proj2 (Hsites slot id Hslot g (sites g)) (conj Hg eq_refl)))
        as [-> [[-> ->] | [-> ->]]]; auto. }
    destruct (refine_graph_function_slots (sites slot) slot (sites slot)
                (proj2 (Hsites slot id Hslot slot (sites slot))
                   (conj Hb eq_refl)))
      as [Hs70 [[Hsite _] | [Hsite _]]].
    + exists [(70%positive, (RefineCounterexample.x10,
                             Or_unknown.Known 20%positive))].
      split; [ left; reflexivity | split ].
      * intros g c code [Hin | []]. injection Hin as <- <- <-.
        subst slot. split; [ exact Hb | symmetry; exact Hsite ].
      * intros g Hg. destruct (Hdom g Hg) as [-> _]. reflexivity.
    + exists [(70%positive, (RefineCounterexample.x11,
                             Or_unknown.Known 20%positive))].
      split; [ right; left; reflexivity | split ].
      * intros g c code [Hin | []]. injection Hin as <- <- <-.
        subst slot. split; [ exact Hb | symmetry; exact Hsite ].
      * intros g Hg. destruct (Hdom g Hg) as [-> _]. reflexivity.
  - exact Henv.
  - exact Hfrag.
  - intros s allocated Hl. discriminate Hl.
  - intros s allocated Hl. discriminate Hl.
  - exact Hcopies.
  - split; [ vm_compute; reflexivity | ].
    intros s k w Hin. vm_compute in Hin. destruct Hin.
  - right. exists RefineCounterexample.x10. split; [ | split ].
    + cbn [rebuilt_value]. exists (fun _ => RefineCounterexample.x10), [].
      split; [ reflexivity | split ].
      { apply may_come_from_self; vm_compute; auto. }
      split.
      * intros f id Hf g b. split.
        -- intros Hcons.
           destruct (refine_graph_function_slots _ _ _ Hcons)
             as [-> [[_ ->] | [Habs _]]];
             [ split; reflexivity | discriminate Habs ].
        -- intros [Hg ->]. unfold is_bound in Hg. cbn [lookup] in Hg.
           destruct (Pos.eqb g 70) eqn:Hg70; [ | discriminate Hg ].
           rewrite (proj1 (Pos.eqb_eq g 70) Hg70).
           cbn. left. reflexivity.
      * apply for_all_bindings_lookup. intros w x Hw f id Hf Hfu.
        exfalso.
        cbn [lookup] in Hw.
        destruct (Pos.eqb w 1) eqn:Hw1; [ | discriminate Hw ].
        rewrite (proj1 (Pos.eqb_eq w 1) Hw1) in Hfu.
        vm_compute in Hfu. discriminate Hfu.
    + intros a (_ & [Has | Hsrc] & _) _;
        [ vm_compute in Has; discriminate Has | ].
      cbn [s_sources solution_of_tables] in Hsrc.
      apply mem_pair_In in Hsrc. cbn in Hsrc.
      destruct Hsrc as [[= Habs] | [[= <-] | []]].
      cbn [describes]. left. vm_compute. reflexivity.
    + intros _. vm_compute. reflexivity.
  - apply type_satisfied_b_iff. vm_compute. reflexivity.
Qed.

Theorem refine_example : refine_example_statement.
Proof. split; vm_compute; reflexivity. Qed.



Theorem rewrite_keeps_null : rewrite_keeps_null_statement.
Proof.
  intros ctx tenv canon statics statics' valuation valuation' fuel m t Henv
    Hcopies Hsat.
  exact (proj1 (rewrite_keeps_null_fuel all_fixes eq_refl ctx tenv canon
                  statics statics' valuation valuation' Henv Hcopies fuel)
           m t Hsat).
Qed.

Theorem null_fix_needed : null_fix_needed_statement.
Proof.
  intros H. destruct (create_valuation empty_static_env)
    as (valuation & Henv & _ & Hcopies).
  pose proof (H NullCounterexample.context Typing_env.create (fun n _ _ => n)
                empty_static_env empty_static_env valuation valuation 5
                NullCounterexample.metadata NullCounterexample.closure_or_null
                Henv Hcopies) as Hsound.
  clear H.
  apply type_satisfied_b_iff in Hsound.
  - vm_compute in Hsound. discriminate Hsound.
  - apply type_satisfied_b_iff. vm_compute. reflexivity.
Qed.

(* ======================================================================== *)
(* An example                                                               *)
(* ======================================================================== *)

Theorem example : example_statement.
Proof. split; vm_compute; reflexivity. Qed.

Theorem result_types_example : result_types_example_statement.
Proof. split; vm_compute; reflexivity. Qed.

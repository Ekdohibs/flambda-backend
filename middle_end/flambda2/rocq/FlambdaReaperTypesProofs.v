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
   ([rewrite_forgets]). The counterexamples are computed. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaPrimitiveSpec
  FlambdaPrimitiveProofs FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperRebuild
  FlambdaReaperSolution
  FlambdaReaperSolutionSpec FlambdaReaperSolutionProofs
  FlambdaReaperFlowSpec FlambdaReaperTypes FlambdaReaperTypesSpec.
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
  - intros (vs & ->). reflexivity.
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
  - destruct Hv as (vs & ->). discriminate.
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
  closure_free_non_null head = true
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
  intros [Hcf Hal]. cbn [closure_free_non_null] in Hcf.
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
  closure_free_blocks blocks = true
  ∧ ∀ (s : simple) (k : flambda_kind) (w : value),
      (s, k) ∈ aliases_of_row_like_for_blocks blocks ->
      eval_simple vars s = Some w -> unallocated w.
Proof.
  intros [Hcf Hal]. cbn [closure_free_non_null] in Hcf.
  apply andb_true_iff in Hcf as [_ Hcf]. split; [ exact Hcf | ].
  intros s k w Hin. apply (Hal s k w).
  cbn [aliases_of_head_of_kind_value_non_null].
  apply in_or_app. right. apply in_or_app. right.
  apply in_or_app. right. apply in_or_app. left. exact Hin.
Qed.

(* The fields of a case of blocks in the fragment are. *)
Lemma block_case_fragment (vars : env) (case : row_like_block_case) :
  forallb closure_free_type (case_maps_to case) = true ->
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
  closure_free_blocks (Row_like_for_blocks known_tags other_tags alloc_mode)
  = true ->
  (∀ (s : simple) (k : flambda_kind) (w : value),
     (s, k)
     ∈ aliases_of_row_like_for_blocks
         (Row_like_for_blocks known_tags other_tags alloc_mode) ->
     eval_simple vars s = Some w -> unallocated w) ->
  (tag, Or_unknown.Known case) ∈ known_tags ->
  ∀ ty : flambda_type, ty ∈ case_maps_to case -> type_in_fragment vars ty.
Proof.
  intros Hcf Hal Hin. cbn [closure_free_blocks] in Hcf.
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
  closure_free_blocks
    (Row_like_for_blocks known_tags (Or_bottom.Ok case) alloc_mode)
  = true ->
  (∀ (s : simple) (k : flambda_kind) (w : value),
     (s, k)
     ∈ aliases_of_row_like_for_blocks
         (Row_like_for_blocks known_tags (Or_bottom.Ok case) alloc_mode) ->
     eval_simple vars s = Some w -> unallocated w) ->
  ∀ ty : flambda_type, ty ∈ case_maps_to case -> type_in_fragment vars ty.
Proof.
  intros Hcf Hal. cbn [closure_free_blocks] in Hcf.
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

(* A type of the fragment has no closure that
   [meet_single_closures_entry] could answer. *)
Lemma meet_single_closures_entry_fragment (t : flambda_type)
    (r : function_slot * Alloc_mode.t * closures_entry * function_type) :
  type_in_fragment vars t ->
  Provers.meet_single_closures_entry tenv t
  = Some (Meet_shortcut.Known_result r) ->
  False.
Proof.
  intros Ht. unfold Provers.meet_single_closures_entry,
    Provers.gen_value_to_meet, Provers.expand_head.
  destruct (expand_head tenv t) as [e | ] eqn:He; cbn; [ | discriminate ].
  pose proof (expand_head_fragment t e He Ht) as [Hcf _].
  destruct e as [k [ | [head | head | k' head | head | head] | ]];
    unfold descr_oub_of_expanded; cbn [Expanded_type.descr];
    try (destruct k as [ | [] | | ]; discriminate);
    try discriminate.
  destruct head as [[ | h | ] is_null]; try discriminate.
  cbn in Hcf.
  destruct h; cbn; discriminate.
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
(* The rewriting is sound                                                   *)
(* ======================================================================== *)

Section Sound.

Variable fx : fixes.
Hypothesis Hfix_boxed : fix_boxed fx = true.
Hypothesis Hfix_alias : fix_alias fx = true.

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
Hypothesis Hlocal : is_local_slots_only is_local.
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

Lemma rule_closure_impossible (m : rewrite_metadata) (t : flambda_type)
    (spec : closure_rewrite) :
  type_in_fragment valuation t ->
  rewrite_rule fx ctx tenv m t = Rule_closure spec -> False.
Proof.
  intros Ht. unfold rewrite_rule. destruct (forgets m); [ discriminate | ].
  destruct (kind t); try discriminate.
  destruct (Provers.meet_single_closures_entry tenv t) as [[r | | ] | ] eqn:Hm;
    try discriminate.
  intros _. exact (meet_single_closures_entry_fragment valuation tenv Hfrag t r
                     Ht Hm).
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
  closure_free_blocks blocks = true
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
  - destruct Hfr as [Hcf _]. discriminate Hcf.
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
    + exfalso. exact (rule_closure_impossible m t spec Ht Hrule).
Qed.

End Sound.

Theorem rewrite_type_sound : rewrite_type_sound_statement.
Proof.
  intros is_local F T sets tenv canon statics statics' valuation valuation'
    fuel m t v v' Hc Hlocal Henv Hfrag Halloc Hst Hcopies Ht Hv Hsat.
  exact (proj1 (rewrite_sound all_fixes eq_refl eq_refl is_local F T sets tenv
                  canon statics statics' valuation valuation' Hc Hlocal Henv
                  Hfrag Halloc Hst Hcopies fuel)
           m t v v' Ht Hv Hsat).
Qed.

Theorem rewrite_symbol_type_sound : rewrite_symbol_type_sound_statement.
Proof.
  intros is_local F T sets tenv canon statics statics' valuation valuation' s
    allocated Hc Hlocal Henv Hfrag Halloc Hst Hcopies Hl.
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
  apply (proj2 (rewrite_sound all_fixes eq_refl eq_refl is_local F T sets tenv
                  canon statics statics' valuation valuation' Hc Hlocal Henv
                  Hfrag Halloc Hst Hcopies (rewrite_fuel tenv ty))
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
    code_var type_var v v' Hc Hlocal Henv Hfrag Halloc Hst Hcopies Hfind Hl
    Hv.
  unfold rewrite_result_type, rewrite_concrete_type_of, kind_of_name.
  destruct (Typing_env.find tenv (Name_var type_var) None) as [ty0 | ] eqn:Hf;
    [ | contradiction ].
  unfold concrete_type_of. rewrite (find_with_kind tenv _ ty0 Hf).
  pose proof (find_fragment valuation tenv Hfrag _ _ ty0 Hf) as Hfr.
  pose proof (find_satisfied_strictly statics valuation tenv (Name_var type_var)
                None ty0 v Henv Hf Hl) as Hsat.
  pose proof (metadata_for_name_at (plain_context is_local F T sets)
                (Node_variable code_var)) as [Hat Hhu].
  apply (proj2 (rewrite_sound all_fixes eq_refl eq_refl is_local F T sets tenv
                  canon statics statics' valuation valuation' Hc Hlocal Henv
                  Hfrag Halloc Hst Hcopies (rewrite_fuel tenv ty0))
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
    valuation' n s k vs hp rd sym allocated Hlocal Hcheck Hrel Hrun Hstep
    Henv Hfrag Hcopies Hl.
  destruct (Hrel n s k vs hp rd Hrun Hstep)
    as [Halloc (m & t & vs' & rd' & Hrun' & Hstep' & Hst)].
  exists m, t, vs', rd'. split; [ exact Hrun' | split; [ exact Hstep' | ] ].
  exact (rewrite_symbol_type_sound is_local (traverse_unit flag u e) T sets
           tenv canon (state_statics s) (state_statics t) valuation
           valuation' sym allocated
           (check_closed_sound is_local _ T Hcheck) Hlocal Henv Hfrag Halloc
           Hst Hcopies Hl).
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
  - apply check_closed_sound. vm_compute. reflexivity.
  - intros f Hf. discriminate Hf.
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
  - apply check_closed_sound. vm_compute. reflexivity.
  - intros f Hf. discriminate Hf.
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
  - apply check_closed_sound. vm_compute. reflexivity.
  - intros f Hf. discriminate Hf.
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

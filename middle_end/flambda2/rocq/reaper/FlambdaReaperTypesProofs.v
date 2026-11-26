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
   [rewrite] together ([rewrite_sound]), for the fixes of [fixes_sound],
   with either fix of the aliases: at a field that its allocation does
   not use, [follow_field] gives [At_least_one_source_no_usages]
   ([follow_field_unused]), which [fix_deleted_field] forgets, and under
   which, with [fix_alias], the rewriting gives a type that every value
   of the kind has ([rewrite_forgets]). The closure case
   ([closure_sound]) reads the closure off the head of its type
   ([meet_closures_head], [closure_of_type]) and off the rebuild
   ([closure_rebuilt_view]); the
   set of the closure is among those that [refine_sets_of_closures]
   keeps ([refine_keeps]: in a least solution an allocation is a source
   of each usage of it that has not any source,
   [usages_carry_sources]); the metadata of its value slots holds where
   they are ([follow_field_for_set_at]), and the join of the slots of
   the sets kept describes what that of the set does
   ([fold_join_covered]). The induction relates a value to its rebuild
   as [rebuilt_under] does, or, for a closure, by the allocation of the
   closure that the metadata describes, its set rebuilt with it
   ([closure_under], [rebuilt_ext]): so is the closure of another slot
   of the set, under the metadata of that slot, whose closure type need
   not be an alias in the fragment of [loose]. The counterexamples and
   the example of a change of representation are computed. The subkinds
   that the reaper with types keeps are proved last, by induction on the
   subkind ([rewrite_kind_with_subkind_sound]).

   Of a least solution, the proofs read that it is closed, has no stale
   tuple, and two facts of its derivations ([usages_carry_sources],
   [has_source_sources]); these hold as well of a least solution of the
   rules with the fix of local fields, by the same induction on the
   derivations ([least_either]), so that each statement is proved of the
   two. The runs keep the static sets whole
   ([rebuild_exit_relation_whole_holds]): the rebuild keeps the sets of
   a term well-formed ([rebuild_sets_well_formed]), so that the rebuilt
   unit binds the symbols of its static sets together
   ([rebuilt_sets_bound], from FlambdaClosuresProofs.sets_inv), as the
   unit does; and the simulation binds in the second run each symbol of
   a static set that has a use ([statics_rel_whole]).

   The result types of an unboxed variable of code
   ([rewrite_unboxed_result_type_sound]) are proved level by level of
   its unboxing ([leaves_sound]): at a level, a value of the first run
   comes from the one allocation of the level ([level_ok]), a field
   that the uses read is used by that allocation and its metadata holds
   where it is held ([level_field_used], [level_metadata_at]), and the
   type at a field of the type of a value holds of the field
   ([destructure_field_sound]). The fields unboxed in turn are those
   that the analysis unboxes ([mk_shape], [mk_unboxed_fields_nested]),
   read into names that it unboxes ([unboxed_fields_of_names_has]),
   whose uses [check_unboxed_uses] keeps from reaching closures
   ([add_usages_of_unboxed]).

   The closure types of a set whose representation changes
   ([rewrite_changed_type_sound]) are proved by induction on the fuel
   of [rewrite_arbitrary_u] ([rewrite_sound_u]), for a relation of the
   first run to the third ([rel_u]): a value that does not change, as
   [rebuilt_ext] relates it to the second run, which the third shares,
   at a node or a set none of whose allocations changes
   ([unchanged_ext]); or a closure whose set changes, rebuilt with its
   set, and the closure in the representation of its allocation
   ([changed_ext]). The cases of [rewrite] are those of [rewrite_sound]
   for that relation ([identity_sound_u], [closure_sound_u],
   [closure_rebuilt_view_u]): what a block or a value slot holds does
   not change ([changed_unstored]), and the rule of a closure keeps no
   set that changes ([rewrite_rule_u]), so that the closures of the set
   at the other slots do not change either. The case of a set that
   changes ([changed_sound]): the set kept is that of the closure
   ([refine_keeps]), with the representation of its allocation; the
   new function slots are those of the set, renamed injectively
   ([rename_inj]), with their code ids; the closure at another slot is
   the closure of the third run at its new slot, of the representation
   of its allocation ([reprs_of_sets]), so that the induction applies
   to it.

   The new value slots ([rewrite_changed_type_sound_slots],
   [changed_slots_sound]): the leaves of [changed_leaves] are those of
   the fields of the representation, in order
   ([changed_field_leaves_paths]), a field of the representation being
   read from the closures of the set, so that the classification of the
   fields read is not empty ([repr_leaves_ok]). A value slot read is
   used by an allocation of the set ([gfoc_used]), so that the second run
   has it, related, at the node that the set holds it at, where the
   metadata of [pattern_metadata] holds ([gfoc_metadata_at], as
   [follow_field_for_set_at]: the queries of FlambdaReaperUnboxing.v are
   those of FlambdaReaperTypes.v, [field_used_top_agree],
   [field_used_as_agree]). A value slot unboxed in turn is a level of
   [leaves_sound_with], the leaves of the result types for a rewriting
   sound of what a field other than a function slot holds
   ([gfoc_level]). A slot that the first run has not is of the unknown
   type at every path ([destructure_path_unknown]). The representations
   of the unit meet [repr_leaves_ok] ([unit_reprs_leaves_ok]): the name
   a representation is given for is the allocation that dominates it
   (FlambdaReaperUnboxingProofs.check_dominated_sound), held at a
   function slot by the closure of [tcr] whose set gives it, which is of
   its set ([function_slots_mutual]), so that the two read the fields of
   the same closures ([gfoc_ext], [gfoc_spec]). *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaPrimitiveSpec
  FlambdaPrimitiveProofs FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperRebuild
  FlambdaReaperSolution
  FlambdaReaperSolutionSpec FlambdaReaperSolutionProofs
  FlambdaReaperFlowSpec FlambdaReaperLeast FlambdaReaperLeastSpec
  FlambdaReaperLeastProofs FlambdaReaperLocalFields FlambdaReaperTypes
  FlambdaReaperTypesSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeMeaningSpec FlambdaTypeMeaningProofs
  FlambdaBindingTime FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaTypingEnvProofs FlambdaExpandHead FlambdaExpandHeadSpec
  FlambdaExpandHeadProofs FlambdaTypeAccessSpec FlambdaProvers
  FlambdaTypeMeaningBoolSpec FlambdaTypeMeaningBoolProofs
  FlambdaEnvAliases FlambdaEnvAliasesSpec FlambdaEnvAliasesProofs.
From OxCamlFlambda2 Require FlambdaReaperUnboxing
  FlambdaReaperUnboxingRefinesCallsSpec FlambdaReaperUnboxingRefinesProofs
  FlambdaReaperUnboxingRefinesCallsSimProofs.
From OxCamlFlambda2 Require FlambdaReaperUnboxingFuelSpec
  FlambdaReaperUnboxingFuelProofs.
Import ListNotations.

Module UF := FlambdaReaperUnboxingFuelSpec.
Module UFP := FlambdaReaperUnboxingFuelProofs.

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

(* A least solution of the rules before the fix of local fields, or of
   those with it: what the proofs read of a least solution holds of
   both. *)
Definition least_either (is_local : field -> bool) (F : facts) (T : tables)
    : Prop :=
  least_solution is_local F T ∨ least_solution_local is_local F T.

Section Least_facts.

Variable is_local : field -> bool.
Variable F : facts.
Variable T : tables.
Hypothesis Hleast : least_either is_local F T.

Local Abbreviation G := (graph_of_facts F).
Local Abbreviation sol := (solution_of_tables T).

Lemma least_closed_tables : closed is_local G sol.
Proof.
  destruct Hleast as [Hl | Hl];
    [ exact (least_closed is_local F T Hl)
    | exact (least_local_closed is_local F T Hl) ].
Qed.

Lemma usages_not_any (x u : node) :
  s_usages sol x u = true -> s_any_usage sol x = false.
Proof.
  destruct Hleast as [Hl | Hl];
    [ exact (proj1 (least_no_stale is_local F T x u Hl)) | ].
  intros H.
  pose proof (least_local_derivable is_local F T Hl (Atom_usages x u) H)
    as Hd.
  inversion Hd; subst; assumption.
Qed.

Lemma sources_not_any (x s : node) :
  s_sources sol x s = true -> s_any_source sol x = false.
Proof.
  destruct Hleast as [Hl | Hl];
    [ exact (proj2 (least_no_stale is_local F T x s Hl)) | ].
  intros H.
  pose proof (least_local_derivable is_local F T Hl (Atom_sources x s) H)
    as Hd.
  inversion Hd; subst; assumption.
Qed.

(* What may come to a node may come to each of its usages, which have
   any source if the node has. *)
Lemma usages_carry_sources (x u : node) :
  s_usages sol x u = true ->
  (∀ s : node,
     s_sources sol x s = true ->
     s_sources sol u s = true ∨ s_any_source sol u = true)
  ∧ (s_any_source sol x = true -> s_any_source sol u = true).
Proof.
  intros H. destruct Hleast as [Hl | Hl].
  - pose proof (least_derivable is_local F T Hl (Atom_usages x u) H) as Hd.
    clear H. remember (Atom_usages x u) as atom eqn:Ha. revert x u Ha.
    induction Hd; intros x0 u0 Ha; try discriminate Ha.
    + injection Ha as <- <-. split; [ intros s Hs; left; exact Hs | auto ].
    + injection Ha as <- <-. split; [ intros s Hs; left; exact Hs | auto ].
    + injection Ha as <- <-.
      destruct (IHHd1 to_ usage eq_refl) as [IHs IHa].
      pose proof (least_holds is_local F T Hl _ Hd2) as Halias.
      cbn [holds] in Halias. split.
      * intros s Hs.
        destruct (s_any_source sol to_) eqn:Hat;
          [ right; exact (IHa eq_refl) | ].
        apply IHs.
        exact (sources_alias _ _ _ least_closed_tables from to_ s Hs
                 (sources_not_any from s Hs) Halias Hat).
      * intros Has. apply IHa.
        exact (any_source_alias _ _ _ least_closed_tables from to_ Halias
                 Has).
  - pose proof (least_local_derivable is_local F T Hl (Atom_usages x u) H)
      as Hd.
    clear H. remember (Atom_usages x u) as atom eqn:Ha. revert x u Ha.
    induction Hd; intros x0 u0 Ha; try discriminate Ha.
    + injection Ha as <- <-. split; [ intros s Hs; left; exact Hs | auto ].
    + injection Ha as <- <-. split; [ intros s Hs; left; exact Hs | auto ].
    + injection Ha as <- <-.
      destruct (IHHd1 to_ usage eq_refl) as [IHs IHa].
      pose proof (least_local_holds is_local F T Hl _ Hd2) as Halias.
      cbn [holds] in Halias. split.
      * intros s Hs.
        destruct (s_any_source sol to_) eqn:Hat;
          [ right; exact (IHa eq_refl) | ].
        apply IHs.
        exact (sources_alias _ _ _ least_closed_tables from to_ s Hs
                 (sources_not_any from s Hs) Halias Hat).
      * intros Has. apply IHa.
        exact (any_source_alias _ _ _ least_closed_tables from to_ Halias
                 Has).
Qed.

(* A node that has a source has any source, or a source that has a
   source. *)
Lemma has_source_sources (x : node) :
  s_has_source sol x = true ->
  s_any_source sol x = true
  ∨ ∃ y : node, s_sources sol x y = true ∧ s_has_source sol y = true.
Proof.
  intros H. destruct Hleast as [Hl | Hl].
  - pose proof (least_derivable is_local F T Hl (Atom_has_source x) H)
      as Hd.
    clear H. remember (Atom_has_source x) as atom eqn:Ha. revert x Ha.
    induction Hd; intros x0 Ha; try discriminate Ha; injection Ha as <-.
    + left. exact (least_holds is_local F T Hl _ Hd).
    + destruct (s_any_source sol to_) eqn:Hat; [ left; reflexivity | right ].
      pose proof (least_holds is_local F T Hl _ Hd2) as Halias.
      cbn [holds] in Halias.
      destruct (IHHd1 from eq_refl) as [Haf | (y & Hy & Hhy)].
      * rewrite (any_source_alias _ _ _ least_closed_tables from to_ Halias
                   Haf)
          in Hat.
        discriminate Hat.
      * exists y. split; [ | exact Hhy ].
        exact (sources_alias _ _ _ least_closed_tables from to_ y Hy
                 (sources_not_any from y Hy) Halias Hat).
    + destruct (s_any_source sol base) eqn:Hab; [ left; reflexivity | right ].
      exists base. split.
      * exact (sources_constructor _ _ _ least_closed_tables from relation
                 base H Hab).
      * exact (least_holds is_local F T Hl (Atom_has_source base)
                 (by_has_source_constructor _ _ _ from relation base Hd H)).
    + destruct (s_any_source sol base) eqn:Hab; [ left; reflexivity | right ].
      exists base. split.
      * exact (sources_constructor _ _ _ least_closed_tables from relation
                 base H Hab).
      * exact (has_source_function_slot _ _ _ least_closed_tables from
                 relation base H H0).
    + destruct (s_any_source sol base) eqn:Hab; [ left; reflexivity | right ].
      exists base. split.
      * exact (sources_parameter _ _ _ least_closed_tables to_ relation base
                 H Hab).
      * exact (has_source_parameter _ _ _ least_closed_tables to_ relation
                 base (least_holds is_local F T Hl _ Hd) H).
  - pose proof (least_local_derivable is_local F T Hl (Atom_has_source x) H)
      as Hd.
    clear H. remember (Atom_has_source x) as atom eqn:Ha. revert x Ha.
    induction Hd; intros x0 Ha; try discriminate Ha; injection Ha as <-.
    + left. exact (least_local_holds is_local F T Hl _ Hd).
    + destruct (s_any_source sol to_) eqn:Hat; [ left; reflexivity | right ].
      pose proof (least_local_holds is_local F T Hl _ Hd2) as Halias.
      cbn [holds] in Halias.
      destruct (IHHd1 from eq_refl) as [Haf | (y & Hy & Hhy)].
      * rewrite (any_source_alias _ _ _ least_closed_tables from to_ Halias
                   Haf)
          in Hat.
        discriminate Hat.
      * exists y. split; [ | exact Hhy ].
        exact (sources_alias _ _ _ least_closed_tables from to_ y Hy
                 (sources_not_any from y Hy) Halias Hat).
    + destruct (s_any_source sol base) eqn:Hab; [ left; reflexivity | right ].
      exists base. split.
      * exact (sources_constructor _ _ _ least_closed_tables from relation
                 base H Hab).
      * exact (least_local_holds is_local F T Hl (Atom_has_source base)
                 (byl_has_source_constructor _ _ _ from relation base Hd H)).
    + destruct (s_any_source sol base) eqn:Hab; [ left; reflexivity | right ].
      exists base. split.
      * exact (sources_constructor _ _ _ least_closed_tables from relation
                 base H Hab).
      * exact (has_source_function_slot _ _ _ least_closed_tables from
                 relation base H H0).
    + destruct (s_any_source sol base) eqn:Hab; [ left; reflexivity | right ].
      exists base. split.
      * exact (sources_parameter _ _ _ least_closed_tables to_ relation base
                 H Hab).
      * exact (has_source_parameter _ _ _ least_closed_tables to_ relation
                 base (least_local_holds is_local F T Hl _ Hd) H).
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

(* A field that the allocation does not use: [follow_field] gives
   [At_least_one_source_no_usages]. *)
Lemma follow_field_unused (m : rewrite_metadata) (a from : node)
    (f : field) :
  describes sol m a -> forgets m = false -> g_constructor G a f from ->
  is_local f = false -> s_field_used sol a f = false ->
  follow_field ctx m f = At_least_one_source_no_usages.
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

(* So [follow_field] forgets. *)
Lemma follow_field_forgets (m : rewrite_metadata) (a from : node)
    (f : field) :
  describes sol m a -> forgets m = false -> g_constructor G a f from ->
  is_local f = false -> s_field_used sol a f = false ->
  forgets (follow_field ctx m f) = true.
Proof.
  intros Hd Hm Hcons Hloc Hfu.
  rewrite (follow_field_unused m a from f Hd Hm Hcons Hloc Hfu).
  reflexivity.
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
      lookup g (snd acc) = Some p ->
      ∃ pe : rewrite_metadata * bool,
        lookup g (snd e) = Some pe
        ∧ ∀ a : node, describes sol (fst pe) a -> describes sol (fst p) a.

Lemma covered_refl (e : slots_metadata) : slots_covered e e.
Proof.
  split.
  - intros w mw Hw. exists mw. auto.
  - intros g p Hg. exists p. auto.
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
    apply lookup_inter_slots in Hg as (pa & pb & Ha & _ & ->).
    destruct (Hf g pa Ha) as (pe & Hpe & Hd). exists pe.
    split; [ exact Hpe | ].
    intros a H. cbn [fst]. apply join_metadata_describes. left. exact (Hd a H).
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
    apply lookup_inter_slots in Hg as (pa & pb & _ & Hb & ->).
    exists pb. split; [ exact Hb | ].
    intros a H. cbn [fst]. apply join_metadata_describes. right. exact H.
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

Lemma unknown_fragment {loose : bool} (vars : env) (k : flambda_kind) :
  type_in_fragment_with loose vars (unknown k).
Proof.
  split; [ destruct k as [ | [] | | ]; reflexivity | ].
  intros s k' w Hin. destruct k as [ | [] | | ]; destruct Hin.
Qed.

Lemma bottom_fragment {loose : bool} (vars : env) (k : flambda_kind) :
  type_in_fragment_with loose vars (bottom k).
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
Definition non_null_in_fragment (loose : bool) (vars : env)
    (head : head_of_kind_value_non_null) : Prop :=
  shaped_non_null_with loose head = true
  ∧ ∀ (s : simple) (k : flambda_kind) (w : value),
      (s, k) ∈ aliases_of_head_of_kind_value_non_null head ->
      eval_simple vars s = Some w -> unallocated w.

Lemma head_fragment {loose : bool} (vars : env)
    (head : head_of_kind_value_non_null) (is_null : is_null) :
  type_in_fragment_with loose vars
    (Type_value (Ok (No_alias (Head_of_kind_value (Ok head) is_null)))) ->
  non_null_in_fragment loose vars head.
Proof.
  intros [Hcf Hal]. split; [ exact Hcf | ].
  intros s k w Hin. apply (Hal s k w). cbn [aliases_with_kinds
    aliases_of_descr aliases_of_head_of_kind_value].
  apply in_or_app. left. exact Hin.
Qed.

Lemma immediates_fragment {loose : bool} (vars : env)
    (is_int : option variable)
    (ty : flambda_type) (get_tag : option variable)
    (blocks : Or_unknown.t row_like_for_blocks)
    (extensions : variant_extensions) (is_unique : bool) :
  non_null_in_fragment loose vars
    (Variant is_int (Or_unknown.Known ty) get_tag blocks extensions
       is_unique) ->
  type_in_fragment_with loose vars ty.
Proof.
  intros [Hcf Hal]. cbn [shaped_non_null_with] in Hcf.
  apply andb_true_iff in Hcf as [Hcf _]. split; [ exact Hcf | ].
  intros s k w Hin. apply (Hal s k w).
  cbn [aliases_of_head_of_kind_value_non_null].
  apply in_or_app. right. apply in_or_app. left. exact Hin.
Qed.

Lemma blocks_fragment {loose : bool} (vars : env) (is_int : option variable)
    (immediates : Or_unknown.t flambda_type) (get_tag : option variable)
    (blocks : row_like_for_blocks) (extensions : variant_extensions)
    (is_unique : bool) :
  non_null_in_fragment loose vars
    (Variant is_int immediates get_tag (Or_unknown.Known blocks) extensions
       is_unique) ->
  shaped_blocks_with loose blocks = true
  ∧ ∀ (s : simple) (k : flambda_kind) (w : value),
      (s, k) ∈ aliases_of_row_like_for_blocks blocks ->
      eval_simple vars s = Some w -> unallocated w.
Proof.
  intros [Hcf Hal]. cbn [shaped_non_null_with] in Hcf.
  apply andb_true_iff in Hcf as [_ Hcf]. split; [ exact Hcf | ].
  intros s k w Hin. apply (Hal s k w).
  cbn [aliases_of_head_of_kind_value_non_null].
  apply in_or_app. right. apply in_or_app. right.
  apply in_or_app. right. apply in_or_app. left. exact Hin.
Qed.

(* The fields of a case of blocks in the fragment are. *)
Lemma block_case_fragment {loose : bool} (vars : env)
    (case : row_like_block_case) :
  forallb (shaped_type_with loose) (case_maps_to case) = true ->
  (∀ (s : simple) (k : flambda_kind) (w : value),
     (s, k) ∈ flat_map aliases_with_kinds (case_maps_to case) ->
     eval_simple vars s = Some w -> unallocated w) ->
  ∀ ty : flambda_type,
    ty ∈ case_maps_to case -> type_in_fragment_with loose vars ty.
Proof.
  intros Hcf Hal ty Hty. split.
  - rewrite forallb_forall in Hcf. exact (Hcf ty Hty).
  - intros s k w Hin. apply (Hal s k w). apply in_flat_map.
    exists ty. auto.
Qed.

Lemma known_case_fragment {loose : bool} (vars : env)
    (known_tags : Keyed.finite_map Z (Or_unknown.t row_like_block_case))
    (other_tags : Or_bottom.t row_like_block_case) (alloc_mode : Alloc_mode.t)
    (tag : Z) (case : row_like_block_case) :
  shaped_blocks_with loose
    (Row_like_for_blocks known_tags other_tags alloc_mode)
  = true ->
  (∀ (s : simple) (k : flambda_kind) (w : value),
     (s, k)
     ∈ aliases_of_row_like_for_blocks
         (Row_like_for_blocks known_tags other_tags alloc_mode) ->
     eval_simple vars s = Some w -> unallocated w) ->
  (tag, Or_unknown.Known case) ∈ known_tags ->
  ∀ ty : flambda_type,
    ty ∈ case_maps_to case -> type_in_fragment_with loose vars ty.
Proof.
  intros Hcf Hal Hin. cbn [shaped_blocks_with] in Hcf.
  apply andb_true_iff in Hcf as [Hcf _]. rewrite forallb_forall in Hcf.
  apply block_case_fragment.
  - exact (Hcf _ Hin).
  - intros s k w Hs. apply (Hal s k w).
    cbn [aliases_of_row_like_for_blocks]. apply in_or_app. left.
    apply in_flat_map. exists (tag, Or_unknown.Known case).
    split; [ exact Hin | ]. cbn [snd aliases_of_block_case].
    apply in_or_app. left. exact Hs.
Qed.

Lemma other_case_fragment {loose : bool} (vars : env)
    (known_tags : Keyed.finite_map Z (Or_unknown.t row_like_block_case))
    (case : row_like_block_case) (alloc_mode : Alloc_mode.t) :
  shaped_blocks_with loose
    (Row_like_for_blocks known_tags (Or_bottom.Ok case) alloc_mode)
  = true ->
  (∀ (s : simple) (k : flambda_kind) (w : value),
     (s, k)
     ∈ aliases_of_row_like_for_blocks
         (Row_like_for_blocks known_tags (Or_bottom.Ok case) alloc_mode) ->
     eval_simple vars s = Some w -> unallocated w) ->
  ∀ ty : flambda_type,
    ty ∈ case_maps_to case -> type_in_fragment_with loose vars ty.
Proof.
  intros Hcf Hal. cbn [shaped_blocks_with] in Hcf.
  apply andb_true_iff in Hcf as [_ Hcf].
  apply block_case_fragment; [ exact Hcf | ].
  intros s k w Hs. apply (Hal s k w).
  cbn [aliases_of_row_like_for_blocks]. apply in_or_app. right.
  cbn [aliases_of_block_case]. apply in_or_app. left. exact Hs.
Qed.

Lemma contents_fragment {loose : bool} (vars : env) (k : boxable_number)
    (contents : flambda_type) (alloc_mode : Alloc_mode.t) :
  non_null_in_fragment loose vars (Boxed_number k contents alloc_mode) ->
  type_in_fragment_with loose vars contents.
Proof.
  intros [Hcf Hal]. split; [ exact Hcf | ].
  intros s k' w Hin. exact (Hal s k' w Hin).
Qed.

(* A type that is not an alias, expanded, is itself or unknown or
   bottom. *)
Lemma of_non_alias_fragment {loose : bool} (vars : env) (ty : flambda_type)
    (e : Expanded_type.t) :
  Expanded_type.of_non_alias_type ty = Some e ->
  type_in_fragment_with loose vars ty ->
  type_in_fragment_with loose vars (Expanded_type.to_type e).
Proof.
  destruct ty as [d | d | k d | d | d]; destruct d as [ | [h | s] | ];
    cbn; intros He; try discriminate; injection He as <-; cbn;
    intros Hfrag; try destruct k;
    first [ exact Hfrag
          | split; [ reflexivity | intros s' k' w Hin; cbn in Hin;
                                   destruct Hin ] ].
Qed.

(* The expanded type of a constant. *)
Lemma create_const_fragment {loose : bool} (vars : env) (c : const) :
  type_in_fragment_with loose vars
    (Expanded_type.to_type (Expanded_type.create_const c)).
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

Context {loose : bool}.
Variable vars : env.
Variable tenv : Typing_env.t.

Hypothesis Hfrag : env_in_fragment_with loose vars tenv.

Lemma any_non_null_value_fragment :
  type_in_fragment_with loose vars any_non_null_value.
Proof.
  split; [ reflexivity | ]. intros s k w Hin. cbn in Hin. destruct Hin.
Qed.

(* What the environment has for a name is of the fragment. *)
Lemma find_fragment (n : name) (k : option flambda_kind) (ty : flambda_type) :
  Typing_env.find tenv n k = Some ty -> type_in_fragment_with loose vars ty.
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
  expand_head tenv t = Some e -> type_in_fragment_with loose vars t ->
  type_in_fragment_with loose vars (Expanded_type.to_type e).
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

Lemma type_for_const_fragment {loose : bool} (vars : env) (c : const) :
  type_in_fragment_with loose vars (type_for_const c).
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
Lemma alias_shaped {loose : bool} (t : flambda_type) :
  is_alias t = true -> shaped_type_with loose t = true.
Proof.
  unfold is_alias.
  destruct t as [d | d | k d | d | d]; destruct d as [ | [h | s] | ];
    cbn; try discriminate; reflexivity.
Qed.

(* A case of closures of a type of the fragment: its slots are among those
   of its index, its closure types are aliases unless [loose], and all its
   types are of the fragment. *)
Lemma closures_case_fragment {loose : bool} (vars : env)
    (known : finite_map row_like_closures_case) (alloc_mode : Alloc_mode.t)
    (is_null : is_null) (slot : function_slot) (case : row_like_closures_case)
    (fts : finite_map (Or_unknown.t function_type))
    (cts vsts : finite_map flambda_type) :
  type_in_fragment_with loose vars
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
       (g, ty) ∈ cts ->
       (loose = true ∨ is_alias ty = true)
       ∧ type_in_fragment_with loose vars ty)
  ∧ (∀ (w : value_slot) (ty : flambda_type),
       (w, ty) ∈ vsts -> type_in_fragment_with loose vars ty).
Proof.
  intros [Hsh Hal] Hin Hentry.
  cbn [shaped_type_with shaped_non_null_with shaped_closures_with] in Hsh.
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
    apply andb_true_iff in Hty as [Hla Hty].
    split; [ apply orb_true_iff; exact Hla | split; [ exact Hty | ] ].
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

Lemma lookup_map_slots {B : Type} (h : function_slot -> B)
    (fts : list function_slot) (g : function_slot) (b : B) :
  lookup g (map (fun g' => (g', h g')) fts) = Some b -> b = h g.
Proof.
  induction fts as [ | g' fts IH]; [ discriminate | ].
  cbn [map lookup]. destruct (Pos.eqb_spec g g') as [-> | _];
    [ intros [= <-]; reflexivity | exact IH ].
Qed.

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
    ∧ map fst fsl = fts
    ∧ ∀ (g : function_slot) (p : rewrite_metadata * bool),
        lookup g fsl = Some p -> fst p = Single_source (closure_at set g).
Proof.
  eexists. split; [ reflexivity | split ].
  - rewrite map_map. cbn [fst]. apply map_id.
  - intros g p Hg. apply lookup_map_slots in Hg. rewrite Hg. reflexivity.
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
Hypothesis Hfix_fields : fix_alias fx = true ∨ fix_deleted_field fx = true.
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
Variable loose : bool.

Local Abbreviation ctx := (plain_context is_local F T sets).
Local Abbreviation G := (graph_of_facts F).
Local Abbreviation sol := (solution_of_tables T).

Hypothesis Hc : closed is_local G sol.
Hypothesis Hleast : least_either is_local F T.
Hypothesis Hlocal : is_local_slots_only is_local.
Hypothesis Hsets : sets_of_graph G sets.
Hypothesis Henv : typing_env_satisfied_strictly statics valuation tenv.
Hypothesis Hfrag : env_in_fragment_with loose valuation tenv.
Hypothesis Halloc : statics_allocated sol statics.
Hypothesis Hst : statics_rebuilt G sol statics statics'.
Hypothesis Hcopies : copies_valued tenv canon valuation valuation'.
Hypothesis Hwhole : loose = true -> static_sets_whole sol statics statics'.

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

(* ------------------------------------------------------------------------ *)
(* Closures rebuilt with their set                                         *)
(* ------------------------------------------------------------------------ *)

(* A closure of a set and what the second run has for it: of a set made
   by a [Let], the closure at [slot] in both runs; of a static set that
   both runs bind whole, its symbol at [slot]. *)
Definition set_closure_pair (slot : function_slot)
    (fdecls : list (function_slot * code_id))
    (vs vs' : list (value_slot * value)) (v v' : value) : Prop :=
  (v = Value_closure_of_set slot fdecls vs
   ∧ v' = Value_closure_of_set slot fdecls vs')
  ∨ ∃ (csyms : list (function_slot * symbol)) (s : symbol),
      v = Value_symbol s ∧ v' = Value_symbol s
      ∧ lookup slot csyms = Some s
      ∧ ∀ (g : function_slot) (s_g : symbol),
          lookup g csyms = Some s_g ->
          lookup s_g (static_values statics)
          = Some (Static_closure_of_set g csyms fdecls vs)
          ∧ lookup s_g (static_values statics')
            = Some (Static_closure_of_set g csyms fdecls vs').

(* A closure rebuilt with its set, under a metadata that describes the
   allocation of the closure: what [rebuilt_under] says of a closure, with
   no node, which might have no use or other sources. The closure of
   another function slot of the set is rebuilt so under the metadata of
   that slot. *)
Definition closure_under (m : rewrite_metadata) (v v' : value) : Prop :=
  ∃ (slot : function_slot) (fdecls : list (function_slot * code_id))
    (vs vs' : list (value_slot * value)) (sites : function_slot -> node),
    set_closure_pair slot fdecls vs vs' v v'
    ∧ is_bound slot fdecls = true
    ∧ set_sites G fdecls sites
    ∧ for_all_bindings
        (value_slot_rebuilt G sol (rebuilt_value G sol) fdecls sites vs') vs
    ∧ describes sol m (sites slot).

Definition rebuilt_ext (m : rewrite_metadata) (v v' : value) : Prop :=
  rebuilt_under G sol m v v' ∨ closure_under m v v'.

Lemma closure_pair_in_set (slot : function_slot)
    (fdecls : list (function_slot * code_id))
    (vs vs' : list (value_slot * value)) (v v' : value) :
  set_closure_pair slot fdecls vs vs' v v' -> is_bound slot fdecls = true ->
  closure_in_set statics v = Some (slot, fdecls, vs)
  ∧ closure_in_set statics' v' = Some (slot, fdecls, vs').
Proof.
  intros Hp Hb. destruct (proj1 (is_bound_lookup slot fdecls) Hb) as [id Hl].
  destruct Hp as [[-> ->] | (csyms & s & -> & -> & Hs & Hall)].
  - cbn [closure_in_set]. rewrite Hl. split; reflexivity.
  - destruct (Hall slot s Hs) as [H1 H2]. cbn [closure_in_set].
    rewrite H1, H2, Hl. split; reflexivity.
Qed.

Lemma closure_under_in_set (m : rewrite_metadata) (v v' : value) :
  closure_under m v v' ->
  ∃ (slot : function_slot) (fdecls : list (function_slot * code_id))
    (vs vs' : list (value_slot * value)),
    closure_in_set statics v = Some (slot, fdecls, vs)
    ∧ closure_in_set statics' v' = Some (slot, fdecls, vs').
Proof.
  intros (slot & fdecls & vs & vs' & sites & Hp & Hb & _).
  exists slot, fdecls, vs, vs'. exact (closure_pair_in_set _ _ _ _ _ _ Hp Hb).
Qed.

(* A closure is no block, boxed number or string. *)
Lemma closure_not_other (st : static_env) (v : value)
    (p : function_slot * list (function_slot * code_id)
         * list (value_slot * value)) :
  closure_in_set st v = Some p ->
  block_of st v = None ∧ boxed_number_of st v = None ∧ string_of st v = None.
Proof.
  destruct v; cbn [closure_in_set block_of boxed_number_of string_of];
    try discriminate; try (intros _; auto; fail).
  destruct (lookup s (static_values st)) as [[] | ]; try discriminate;
    intros _; auto.
Qed.

Lemma closure_under_kind (m : rewrite_metadata) (v v' : value) :
  closure_under m v v' -> kind_of_value v' = kind_of_value v.
Proof.
  intros (slot & fdecls & vs & vs' & sites & Hp & _).
  destruct Hp as [[-> ->] | (csyms & s & -> & -> & _)]; reflexivity.
Qed.

Lemma closure_under_not_const (m : rewrite_metadata) (c : const) (v' : value)
    : ~ closure_under m (Value_const c) v'.
Proof.
  intros (slot & fdecls & vs & vs' & sites & Hp & _).
  destruct Hp as [[Habs _] | (csyms & s & Habs & _)]; discriminate Habs.
Qed.

Lemma ext_kind (m : rewrite_metadata) (v v' : value) :
  rebuilt_ext m v v' -> kind_of_value v' = kind_of_value v.
Proof.
  intros [Hv | Hv];
    [ exact (rebuilt_under_kind G sol m v v' Hv)
    | exact (closure_under_kind m v v' Hv) ].
Qed.

Lemma ext_const (m : rewrite_metadata) (c : const) (v' : value) :
  rebuilt_ext m (Value_const c) v' -> v' = Value_const c.
Proof.
  intros [Hv | Hv]; [ exact (rebuilt_under_const G sol m c v' Hv) | ].
  destruct (closure_under_not_const m c v' Hv).
Qed.

Lemma ext_of_const (m : rewrite_metadata) (c : const) :
  rebuilt_ext m (Value_const c) (Value_const c).
Proof. left. left. exists c. split; reflexivity. Qed.

Lemma ext_unallocated (m : rewrite_metadata) (v v' : value) :
  unallocated v -> rebuilt_ext m v v' -> v' = v.
Proof.
  intros Hu [Hv | Hv];
    [ exact (rebuilt_under_unallocated G sol m v v' Hu Hv) | ].
  destruct Hv as (slot & fdecls & vs & vs' & sites & Hp & _).
  destruct Hp as [[-> _] | (csyms & s & -> & -> & _)]; [ | reflexivity ].
  destruct Hu as [(c & Hc') | (s & Hs)]; discriminate.
Qed.

Lemma ext_not_null (m : rewrite_metadata) (v v' : value) :
  rebuilt_ext m v v' -> v <> Value_const Null -> v' <> Value_const Null.
Proof.
  intros [Hv | Hv]; [ exact (rebuilt_under_not_null G sol m v v' Hv) | ].
  intros _. destruct Hv as (slot & fdecls & vs & vs' & sites & Hp & _).
  destruct Hp as [[_ ->] | (csyms & s & _ & -> & _)]; discriminate.
Qed.

(* What a rewriting of the arbitrary types is asked: to be sound, and to
   give, under a metadata that forgets, a type that every value of the
   kind has, in the fragment of [loose], with the closures rebuilt with
   their set. *)
Definition rw_sound_ext
    (rw : rewrite_metadata -> flambda_type -> flambda_type) : Prop :=
  ∀ (m : rewrite_metadata) (t : flambda_type) (v v' : value),
    type_in_fragment_with loose valuation t -> rebuilt_ext m v v' ->
    type_satisfied statics valuation v t ->
    type_satisfied statics' valuation' v' (rw m t).

Definition rw_forgets (rw : rewrite_metadata -> flambda_type -> flambda_type)
    : Prop :=
  ∀ (m : rewrite_metadata) (t : flambda_type) (v v' : value),
    forgets m = true -> type_in_fragment_with loose valuation t ->
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

(* Under a metadata that forgets, with [fix_alias]. *)
Lemma rewrite_forgets (Hfix_alias : fix_alias fx = true) (fuel : nat) :
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

(* A type that is not an alias, under a metadata that forgets, with any
   fixes. *)
Lemma rewrite_forgets_not_alias (fuel : nat) (m : rewrite_metadata)
    (t : flambda_type) (v v' : value) :
  forgets m = true -> get_alias_opt t = None ->
  type_satisfied statics valuation v t ->
  kind_of_value v' = kind_of_value v ->
  type_satisfied statics' valuation' v'
    (rewrite_arbitrary fx ctx tenv canon fuel m t).
Proof.
  intros Hm Ha Hv Hk.
  destruct fuel as [ | [ | fuel]]; cbn [rewrite_arbitrary rewrite];
    try rewrite Ha; try rewrite (rule_forgets m t Hm);
    exact (sat_unknown_of_kind v v' t Hv Hk).
Qed.

(* At a field of a block that no use reads: the field is forgotten
   ([fix_deleted_field]), or so is an alias under a metadata that forgets
   ([fix_alias]). *)
Lemma fields_forget (fuel : nat) :
  fix_deleted_field fx = true
  ∨ rw_forgets (rewrite_arbitrary fx ctx tenv canon fuel).
Proof.
  destruct Hfix_fields as [Ha | Hd]; [ right | left; exact Hd ].
  exact (proj1 (rewrite_forgets Ha fuel)).
Qed.

(* A block of the first run, and what the second has for it. *)
Lemma block_rebuilt (m : rewrite_metadata) (v v' : value) (tag : Z)
    (fields : list value) :
  forgets m = false -> rebuilt_ext m v v' ->
  block_of statics v = Some (tag, fields) ->
  ∃ (a : node) (fields' : list value),
    describes sol m a ∧ block_of statics' v' = Some (tag, fields')
    ∧ fields_rebuilt (field_rebuilt G sol (rebuilt_value G sol) a) 0 fields
        fields'.
Proof.
  intros Hm [Hv0 | Hcu] Hblock;
    [ | exfalso; destruct (closure_under_in_set m v v' Hcu)
          as (slot & fdecls & vs & vs' & Hcis & _);
        rewrite (proj1 (closure_not_other _ _ _ Hcis)) in Hblock;
        discriminate Hblock ].
  destruct Hv0 as [(c & -> & ->) | (n & Hv & Hat & Hhu)];
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
  forgets m = false -> rebuilt_ext m v v' ->
  boxed_number_of statics v = Some c -> kind_of_const c = unboxed_kind k ->
  forgets (follow_field ctx m (Field_boxed_number k)) = false ->
  boxed_number_of statics' v' = Some c.
Proof.
  intros Hm [Hv0 | Hcu] Hbox Hkc Hff;
    [ | exfalso; destruct (closure_under_in_set m v v' Hcu)
          as (slot & fdecls & vs & vs' & Hcis & _);
        rewrite (proj1 (proj2 (closure_not_other _ _ _ Hcis))) in Hbox;
        discriminate Hbox ].
  destruct Hv0 as [(c0 & -> & ->) | (n & Hv & Hat & Hhu)];
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
  forgets m = false -> rebuilt_ext m v v' ->
  string_of statics v = Some bytes -> string_of statics' v' = Some bytes.
Proof.
  intros Hm [Hv0 | Hcu] Hstr;
    [ | exfalso; destruct (closure_under_in_set m v v' Hcu)
          as (slot & fdecls & vs & vs' & Hcis & _);
        rewrite (proj2 (proj2 (closure_not_other _ _ _ Hcis))) in Hstr;
        discriminate Hstr ].
  destruct Hv0 as [(c & -> & ->) | (n & Hv & Hat & Hhu)];
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
Hypothesis Hrw : rw_sound_ext rw.
Hypothesis Hrwf : fix_deleted_field fx = true ∨ rw_forgets rw.

(* A field of a block. *)
Lemma field_sound (m : rewrite_metadata) (a : node) (k : nat)
    (field field' : value) (ty : flambda_type) :
  forgets m = false -> describes sol m a ->
  field_rebuilt G sol (rebuilt_value G sol) a k field field' ->
  type_in_fragment_with loose valuation ty ->
  type_satisfied statics valuation field ty ->
  type_satisfied statics' valuation' field' (rewrite_field fx ctx rw m k ty).
Proof.
  intros Hm Hd (from & Hcons & Hused & Hunused) Hfr Hsat.
  assert (Hk : kind_of_value field' = kind_of_value field).
  { destruct (s_field_used sol a (Field_block k (kind_of_value field)));
      [ exact (rebuilt_value_kind G sol _ _ _ (proj2 (Hused eq_refl)))
      | rewrite (Hunused eq_refl); reflexivity ]. }
  unfold rewrite_field, block_slot. cbv zeta.
  destruct (deleted_field fx (follow_field ctx m (Field_block k (kind ty))))
    eqn:Hdel; [ exact (sat_unknown_of_kind field field' ty Hsat Hk) | ].
  rewrite <- (type_satisfied_kind _ _ _ _ Hsat) in Hdel |- *.
  destruct (s_field_used sol a (Field_block k (kind_of_value field)))
    eqn:Hfu.
  - destruct (Hused eq_refl) as [Hhu Hrv].
    apply (Hrw _ ty field field' Hfr); [ | exact Hsat ].
    left. right. exists from.
    split; [ exact Hrv | split; [ | intros _; exact Hhu ] ].
    exact (follow_field_at ctx is_local Hc m a from _ Hd Hm Hcons
             (not_local_block k _) Hfu).
  - rewrite (Hunused eq_refl).
    pose proof (follow_field_unused ctx is_local Hc m a from _ Hd Hm Hcons
                  (not_local_block k _) Hfu) as Hff.
    rewrite Hff in Hdel |- *.
    destruct Hrwf as [Hdf | Hrwf'];
      [ unfold deleted_field in Hdel; rewrite Hdf in Hdel;
        discriminate Hdel | ].
    apply (Hrwf' At_least_one_source_no_usages ty field _ eq_refl Hfr Hsat).
    reflexivity.
Qed.

(* The fields of a block, from the [k]th. *)
Lemma fields_sound (m : rewrite_metadata) (a : node) :
  forgets m = false -> describes sol m a ->
  ∀ (tys : list flambda_type) (k : nat) (fields fields' : list value),
    fields_rebuilt (field_rebuilt G sol (rebuilt_value G sol) a) k fields
      fields' ->
    (∀ ty : flambda_type,
       ty ∈ tys -> type_in_fragment_with loose valuation ty) ->
    fields_satisfied (type_satisfied statics valuation) tys fields ->
    fields_satisfied (type_satisfied statics' valuation')
      (map (fun p => rewrite_field fx ctx rw m (fst p) (snd p))
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
  (∀ ty : flambda_type,
     ty ∈ case_maps_to case -> type_in_fragment_with loose valuation ty) ->
  row_like_block_case_satisfied statics valuation tag fields case ->
  row_like_block_case_satisfied statics' valuation' tag fields'
    (rewrite_block_case fx ctx rw m case).
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
  shaped_blocks_with loose blocks = true
  ∧ (∀ (s : simple) (k : flambda_kind) (w : value),
       (s, k) ∈ aliases_of_row_like_for_blocks blocks ->
       eval_simple valuation s = Some w -> unallocated w) ->
  row_like_for_blocks_satisfied statics valuation tag fields blocks ->
  row_like_for_blocks_satisfied statics' valuation' tag fields'
    (rewrite_row_like_for_blocks fx ctx rw m blocks).
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
  forgets m = false -> non_null_in_fragment loose valuation head ->
  rebuilt_ext m v v' -> v <> Value_const Null ->
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
    + left. rewrite (ext_const m _ v' Hv). exists i.
      split; [ reflexivity | split; [ exact I | split; [ | exact I ] ] ].
      destruct immediates as [ty | ]; [ | exact I ].
      apply (Hrw m ty (Value_const (Naked_immediate i))
               (Value_const (Naked_immediate i))).
      * exact (immediates_fragment valuation is_int ty get_tag blocks
                 extensions is_unique Hfr).
      * apply ext_of_const.
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
    + rewrite (ext_kind m v v' Hv).
      exact (boxed_number_of_kind statics v c Hbox).
    + rewrite boxed_number_satisfied. exists c.
      split; [ exact (boxed_rebuilt m v v' k c Hm Hv Hbox Hkc Hff) | ].
      split; [ exact Hkc | ].
      apply (Hrw m contents (Value_const c) (Value_const c)
               (contents_fragment valuation k contents am Hfr));
        [ apply ext_of_const | exact Hcontents ].
  - cbn [rewrite_head_of_kind_value_non_null].
    destruct closures as [[ | b known]].
    + rewrite closures_satisfied in Hsat.
      destruct Hsat as (slot & fdecls & vs & _ & Hrow).
      rewrite row_like_for_closures_lookup_iff in Hrow.
      destruct Hrow as (case & Hl & _). discriminate Hl.
    + rewrite (ext_kind m v v' Hv).
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
  type_in_fragment_with loose valuation (Type_value (Ok (No_alias head))) ->
  rebuilt_ext m v v' ->
  head_of_kind_value_satisfied statics valuation v head ->
  head_of_kind_value_satisfied statics' valuation' v'
    (rewrite_head_of_kind_value fx ctx rw m head).
Proof.
  intros Hm Hfr Hv Hsat. destruct head as [non_null is_null].
  cbn [rewrite_head_of_kind_value].
  rewrite head_of_kind_value_satisfied_eq in Hsat |- *.
  destruct Hsat as [[Hnull His] | [Hnn [Hpart His]]].
  - left. subst v. rewrite (ext_const m Null v' Hv).
    split; [ reflexivity | ].
    destruct is_null; [ contradiction | exact I ].
  - right. split; [ exact (ext_not_null m v v' Hv Hnn) | ].
    split; [ | destruct is_null; exact I ].
    destruct non_null as [ | h | ]; [ | | contradiction ].
    + rewrite (ext_kind m v v' Hv). exact Hpart.
    + exact (non_null_sound m h v v' Hm (head_fragment valuation h is_null Hfr)
               Hv Hnn Hpart).
Qed.

(* An expanded type. *)
Lemma expanded_sound (m : rewrite_metadata) (e : Expanded_type.t)
    (v v' : value) :
  forgets m = false ->
  type_in_fragment_with loose valuation (Expanded_type.to_type e) ->
  rebuilt_ext m v v' ->
  type_satisfied statics valuation v (Expanded_type.to_type e) ->
  type_satisfied statics' valuation' v' (rewrite_expanded_head fx ctx rw m e).
Proof.
  intros Hm Hfr Hv Hsat.
  destruct e as [k [ | h | ]]; unfold rewrite_expanded_head;
    cbn [Expanded_type.descr Expanded_type.kind Expanded_type.to_type] in *.
  - apply unknown_satisfied. rewrite (ext_kind m v v' Hv).
    exact (proj1 (unknown_satisfied _ _ _ _) Hsat).
  - destruct h as [hv | hi | kn ns | hr | hreg].
    + rewrite type_satisfied_value in Hsat |- *.
      exact (head_value_sound m hv v v' Hm Hfr Hv Hsat).
    + rewrite type_satisfied_naked_immediate in Hsat |- *.
      cbn [descr_satisfied] in Hsat |- *.
      destruct Hsat as (i & -> & Hnum & _).
      rewrite (ext_const m _ v' Hv). exists i.
      split; [ reflexivity | split; [ exact Hnum | ] ].
      intros relation n [].
    + rewrite type_satisfied_naked_number in Hsat |- *.
      cbn [descr_satisfied] in Hsat |- *.
      destruct Hsat as (c & n & -> & Hk & Hn & Hin).
      rewrite (ext_const m _ v' Hv). exists c, n. auto.
    + rewrite type_satisfied_rec_info in Hsat |- *.
      cbn [descr_satisfied] in Hsat |- *.
      rewrite (ext_kind m v v' Hv). exact Hsat.
    + rewrite type_satisfied_region in Hsat |- *.
      cbn [descr_satisfied] in Hsat |- *.
      rewrite (ext_kind m v v' Hv). exact Hsat.
  - exfalso. exact (bottom_satisfied _ _ _ _ Hsat).
Qed.

(* A type rewritten under [Rule.identity]. *)
Lemma identity_sound (m : rewrite_metadata) (t : flambda_type) (v v' : value)
    :
  forgets m = false -> type_in_fragment_with loose valuation t ->
  rebuilt_ext m v v' -> type_satisfied statics valuation v t ->
  type_satisfied statics' valuation' v' (rewrite_identity fx ctx tenv rw m t).
Proof.
  intros Hm Ht Hv Hsat. unfold rewrite_identity.
  destruct (expand_head tenv t) as [e | ] eqn:He;
    [ | exact (sat_unknown_of_kind v v' t Hsat
                 (ext_kind m v v' Hv)) ].
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

(* A closure rebuilt with its set: a closure of the second run of the
   same set, at the same slot, with the value slots that the rebuild
   keeps; the closure of another slot of the set is that of the first run
   if that is a symbol, and is rebuilt with its set under any metadata
   that describes its allocation. *)
Lemma closure_under_view (m : rewrite_metadata) (v v' : value)
    (slot : function_slot) (fdecls : list (function_slot * code_id))
    (vs : list (value_slot * value)) :
  closure_under m v v' ->
  closure_in_set statics v = Some (slot, fdecls, vs) ->
  ∃ (sites : function_slot -> node) (vs' : list (value_slot * value)),
    closure_in_set statics' v' = Some (slot, fdecls, vs')
    ∧ describes sol m (sites slot)
    ∧ set_sites G fdecls sites
    ∧ for_all_bindings
        (value_slot_rebuilt G sol (rebuilt_value G sol) fdecls sites vs') vs
    ∧ ∀ (g : function_slot) (c' : value),
        is_bound g fdecls = true ->
        project_function_slot statics' slot g v' = Some c' ->
        ∃ c : value,
          project_function_slot statics slot g v = Some c
          ∧ (unallocated c -> c' = c)
          ∧ ∀ mg : rewrite_metadata,
              describes sol mg (sites g) -> closure_under mg c c'.
Proof.
  intros (slot0 & fdecls0 & vs0 & vs' & sites & Hp & Hb & Hsites & Hslots
          & Hd) Hcis.
  destruct (closure_pair_in_set _ _ _ _ _ _ Hp Hb) as [Hcis0 Hcis'].
  rewrite Hcis0 in Hcis. injection Hcis as -> -> ->.
  exists sites, vs'.
  split; [ exact Hcis' | split; [ exact Hd | split; [ exact Hsites | ] ] ].
  split; [ exact Hslots | ].
  destruct (proj1 (is_bound_lookup slot fdecls) Hb) as [id Hl].
  intros g c' Hg Hpc'.
  destruct (proj1 (is_bound_lookup g fdecls) Hg) as [idg Hlg].
  destruct Hp as [[-> ->] | (csyms & s & -> & -> & Hs & Hall)].
  - cbn [project_function_slot] in Hpc' |- *.
    rewrite Pos.eqb_refl, Hl, Hlg in Hpc' |- *. cbn in Hpc' |- *.
    injection Hpc' as <-.
    exists (Value_closure_of_set g fdecls vs).
    split; [ reflexivity | split ].
    + intros [(c0 & Hc0) | (s0 & Hs0)]; discriminate.
    + intros mg Hdg. exists g, fdecls, vs, vs', sites.
      split; [ left; split; reflexivity | ].
      split; [ exact Hg | split; [ exact Hsites | split; [ exact Hslots | ] ] ].
      exact Hdg.
  - destruct (Hall slot s Hs) as [H1 H2].
    cbn [project_function_slot] in Hpc' |- *.
    rewrite H1. rewrite H2 in Hpc'.
    rewrite Pos.eqb_refl, Hl, Hlg in Hpc' |- *. cbn in Hpc' |- *.
    destruct (lookup g csyms) as [s_g | ] eqn:Hsg; [ | discriminate Hpc' ].
    cbn in Hpc' |- *. injection Hpc' as <-.
    exists (Value_symbol s_g). split; [ reflexivity | split; [ auto | ] ].
    intros mg Hdg. exists g, fdecls, vs, vs', sites.
    split; [ right; exists csyms, s_g; auto | ].
    split; [ exact Hg | split; [ exact Hsites | split; [ exact Hslots | ] ] ].
    exact Hdg.
Qed.

(* A closure of the first run, rebuilt under a metadata that does not
   forget: as [closure_under_view] has it, but that the closure of another
   slot of the set is rebuilt with its set only in the fragment of
   [loose], and is that of the first run if that is not allocated. *)
Lemma closure_rebuilt_view (m : rewrite_metadata) (v v' : value)
    (slot : function_slot) (fdecls : list (function_slot * code_id))
    (vs : list (value_slot * value)) :
  forgets m = false -> rebuilt_ext m v v' ->
  closure_in_set statics v = Some (slot, fdecls, vs) ->
  ∃ (sites : function_slot -> node) (vs' : list (value_slot * value)),
    closure_in_set statics' v' = Some (slot, fdecls, vs')
    ∧ describes sol m (sites slot)
    ∧ set_sites G fdecls sites
    ∧ for_all_bindings
        (value_slot_rebuilt G sol (rebuilt_value G sol) fdecls sites vs') vs
    ∧ ∀ (g : function_slot) (c' : value),
        is_bound g fdecls = true ->
        project_function_slot statics' slot g v' = Some c' ->
        ∃ c : value,
          project_function_slot statics slot g v = Some c
          ∧ (unallocated c -> c' = c)
          ∧ (loose = true ->
             ∀ mg : rewrite_metadata,
               describes sol mg (sites g) -> closure_under mg c c').
Proof.
  intros Hm [Hv | Hcu] Hcis.
  2:{ destruct (closure_under_view m v v' slot fdecls vs Hcu Hcis)
        as (sites & vs' & Hcis' & Hd & Hsites & Hslots & Hproj).
      exists sites, vs'.
      split; [ exact Hcis' | split; [ exact Hd | split; [ exact Hsites | ] ] ].
      split; [ exact Hslots | ].
      intros g c' Hg Hpc'.
      destruct (Hproj g c' Hg Hpc') as (c & Hpc & Hu & Hcu').
      exists c.
      split; [ exact Hpc | split; [ exact Hu | intros _; exact Hcu' ] ].
  }
  destruct Hv as [(c & -> & ->) | (n & Hv & Hat & Hhu)];
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
    assert (Hb : is_bound slot fdecls = true)
      by (unfold is_bound; rewrite Hl0; reflexivity).
    assert (Hd : describes sol m (sites slot)).
    { apply (Hat (sites slot) Hmcf).
      apply (has_source_function_slot _ _ _ Hc _ _ (sites slot)
               (proj2 (Hsites slot id0 Hl0 slot (sites slot))
                  (conj Hb eq_refl))
               eq_refl). }
    destruct (closure_under_view m _ _ slot fdecls vs
                (ex_intro _ slot (ex_intro _ fdecls (ex_intro _ vs
                   (ex_intro _ vs' (ex_intro _ sites
                      (conj (or_introl (conj eq_refl eq_refl))
                         (conj Hb (conj Hsites (conj Hslots Hd)))))))))
                (ltac:(cbn [closure_in_set]; rewrite Hl0; reflexivity)))
      as (sites' & vs'' & Hcis' & Hd' & Hsites' & Hslots' & Hproj).
    exists sites', vs''.
    split; [ exact Hcis' | split; [ exact Hd' | split; [ exact Hsites' | ] ] ].
    split; [ exact Hslots' | ].
    intros g c' Hg Hpc'. destruct (Hproj g c' Hg Hpc') as (c & Hpc & Hu & Hcu).
    exists c. split; [ exact Hpc | split; [ exact Hu | intros _; exact Hcu ] ].
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
      exists sites, vs'. split.
      { cbn [closure_in_set]. rewrite Hls', Hl0. reflexivity. }
      split.
      { rewrite Hsite. apply (Hat (Node_symbol s) Hmcf).
        exact (proj1 (Halloc s _ Hls)). }
      split; [ exact Hsites | split; [ exact Hslots | ] ].
      intros g c' Hg Hpc'.
      assert (Hpeq : project_function_slot statics' slot g (Value_symbol s)
                     = project_function_slot statics slot g (Value_symbol s)).
      { cbn [project_function_slot]. rewrite Hls, Hls'. reflexivity. }
      rewrite Hpeq in Hpc'. exists c'.
      split; [ exact Hpc' | split; [ auto | ] ].
      intros Hlo mg Hdg.
      destruct (Hwhole Hlo s slot csyms fdecls vs Hls (Hhus Hhu))
        as [Hs (vs'' & Hall)].
      assert (Hvs : vs'' = vs').
      { destruct (Hall slot s Hs) as [_ H2]. rewrite Hls' in H2.
        injection H2 as ->. reflexivity. }
      subst vs''.
      cbn [project_function_slot] in Hpc'. rewrite Hls in Hpc'.
      rewrite Pos.eqb_refl, Hl0 in Hpc'.
      destruct (proj1 (is_bound_lookup g fdecls) Hg) as [idg Hlg].
      rewrite Hlg in Hpc'. cbn in Hpc'.
      destruct (lookup g csyms) as [s_g | ] eqn:Hsg; [ | discriminate Hpc' ].
      cbn in Hpc'. injection Hpc' as <-.
      exists g, fdecls, vs, vs', sites.
      split; [ right; exists csyms, s_g; auto | ].
      split; [ exact Hg | split; [ exact Hsites | split; [ exact Hslots | ] ] ].
      exact Hdg.
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
    is_alias t = true -> type_in_fragment_with loose valuation t ->
    type_satisfied statics valuation w t ->
    type_satisfied statics' valuation' w (rw m t).

Lemma rewrite_arbitrary_alias (fuel : nat) :
  rw_sound_ext (rewrite_arbitrary fx ctx tenv canon fuel) ->
  rw_alias (rewrite_arbitrary fx ctx tenv canon fuel).
Proof.
  intros Hrw m t w Hal Ht Hsat.
  unfold is_alias in Hal.
  destruct (get_alias_opt t) as [s | ] eqn:Ha; [ | discriminate ].
  pose proof (proj1 (get_alias_satisfied _ _ w t s Ha) Hsat) as [Hs Hkt].
  destruct (proj2 Ht s (kind t) w (get_alias_in_aliases t s Ha) Hs)
    as [(c & ->) | (sym & ->)].
  - apply (Hrw m t (Value_const c) (Value_const c) Ht);
      [ apply ext_of_const | exact Hsat ].
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
  rw_sound_ext rw -> rw_alias rw ->
  type_in_fragment_with loose valuation t -> rebuilt_ext m v v' ->
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
  { rewrite (ext_const m Null v' Hv).
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
    as (sites & vs' & Hcis' & Hdesc & Hsites & Hslots & Hproj).
  pose proof (closure_in_set_slot statics v slot fdecls vs Hcis) as Hslot_b.
  destruct (proj1 (is_bound_lookup slot fdecls) Hslot_b) as [id Hlslot].
  destruct (Hsets fdecls sites slot id Hlslot Hsites)
    as (S0 & HS0 & HS0in & HS0b).
  (* The allocation of the closure, which the metadata describes. *)
  assert (Hself : g_constructor G (sites slot) (Field_function_slot slot)
                    (sites slot)).
  { apply (proj2 (Hsites slot id Hlslot slot (sites slot))).
    split; [ exact Hslot_b | reflexivity ]. }
  (* Its set. *)
  set (nodes0 := map (fun p : function_slot * (node * Or_unknown.t code_id) =>
                        (fst p, fst (snd p))) S0).
  assert (Hnodes : ∀ (g : function_slot) (c : node),
             (g, c) ∈ nodes0 -> is_bound g fdecls = true ∧ c = sites g).
  { intros g c Hin. apply in_map_iff in Hin as ([g' [c' code]] & Hp & Hin).
    cbn [fst snd] in Hp. injection Hp as <- <-.
    exact (HS0in g' c' code Hin). }
  assert (HS0slot : is_bound slot S0 = true) by exact (HS0b slot Hslot_b).
  assert (Hat_g : ∀ g : function_slot,
             is_bound g fdecls = true -> closure_at S0 g = sites g).
  { intros g Hg. unfold closure_at.
    destruct (proj1 (is_bound_lookup g S0) (HS0b g Hg)) as [[c code] HlS0].
    rewrite HlS0.
    exact (proj2 (HS0in g c code (lookup_in g (c, code) S0 HlS0))). }
  pose proof (Hat_g slot Hslot_b) as Hat_slot.
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
    as (fsl & Hshape & Hfsl & Hfsl_md).
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
  { intros g p Hg. destruct (Hcov_f g p Hg) as (p0 & Hp0 & _).
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
        destruct (Hcta g tyg (lookup_in g tyg cts Htyg)) as [Hlo Hfrg].
        destruct (Hproj g closure' Hgd Hproj') as (c & Hpc & Hun & Hcu).
        pose proof (Hcts_sat g tyg c Htyg Hpc) as Hsc.
        destruct (is_alias tyg) eqn:Hal.
        * (* An alias: to a constant or a symbol, kept. *)
          unfold is_alias in Hal.
          destruct (get_alias_opt tyg) as [s | ] eqn:Ha; [ | discriminate ].
          destruct (proj1 (get_alias_satisfied _ _ c tyg s Ha) Hsc)
            as [Hs _].
          rewrite (Hun (proj2 Hfrg s (kind tyg) c
                          (get_alias_in_aliases tyg s Ha) Hs)).
          apply (Hrwa mg tyg c); [ | exact Hfrg | exact Hsc ].
          unfold is_alias. rewrite Ha. reflexivity.
        * (* Not an alias: the closure, rebuilt with its set, under the
             metadata of its slot, which describes its allocation. *)
          destruct Hlo as [Hlo | Habs]; [ | discriminate Habs ].
          apply (Hrw mg tyg c closure' Hfrg); [ right | exact Hsc ].
          apply (Hcu Hlo mg).
          destruct (Hcov_f g (mg, erase) Hgf) as (pe & Hpe & Hdp).
          apply Hdp. rewrite (Hfsl_md g pe Hpe), (Hat_g g Hgd).
          reflexivity.
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
        * left. right. exists from.
          split; [ exact Hrv | split; [ exact Hatf | intros _; exact Hhuf ] ].
        * exact (Hvsts_sat w ty x Hty Hx). }
  destruct (fix_null fx && expanded_maybe_null tenv t); [ | exact Hty' ].
  unfold at_least_this_closure, create_closures, non_null_value in Hty' |- *.
  apply maybe_null_of_not_null. exact Hty'.
Qed.

(* The rewriting, by induction on its fuel. *)
Lemma rewrite_sound (fuel : nat) :
  rw_sound_ext (rewrite_arbitrary fx ctx tenv canon fuel)
  ∧ rw_sound_ext (rewrite fx ctx tenv canon fuel).
Proof.
  induction fuel as [ | fuel [IHa IHr]]; split; intros m t v v' Ht Hv Hsat.
  - exact (sat_unknown_of_kind v v' t Hsat (ext_kind m v v' Hv)).
  - exact (sat_unknown_of_kind v v' t Hsat (ext_kind m v v' Hv)).
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
                   (ext_kind m v v' Hv)).
    + destruct (fix_alias fx && forgets m);
        [ exact (sat_unknown_of_kind v v' t Hsat
                   (ext_kind m v v' Hv)) | ].
      rewrite (ext_unallocated m v v' Hunalloc Hv).
      apply alias_satisfied. split; [ | exact Hkt ].
      change (value_of_name valuation' (canon (Name_var x) (kind t) m)
              = Some v).
      rewrite (Hcopies s (Name_var x) (kind t) m Hcan). exact Hs.
    + destruct (fix_alias fx && forgets m);
        [ exact (sat_unknown_of_kind v v' t Hsat
                   (ext_kind m v v' Hv)) | ].
      rewrite (ext_unallocated m v v' Hunalloc Hv).
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
               (fields_forget fuel) m t v v'
               (rule_identity_not_forgets m t Hrule) Ht Hv Hsat).
    + exact (sat_unknown_of_kind v v' t Hsat
               (ext_kind m v v' Hv)).
    + exact (closure_sound (rewrite_arbitrary fx ctx tenv canon fuel) m t v
               v' spec IHa (rewrite_arbitrary_alias fuel IHa) Ht Hv Hsat
               Hrule).
Qed.

End Sound.

(* The fixes under which the rewriting is proved sound: those of boxed
   numbers, of [refine_sets_of_closures] and of the nullability of
   closure types, and one of the two of the aliases. *)
Definition fixes_sound (fx : fixes) : Prop :=
  fix_boxed fx = true
  ∧ (fix_alias fx = true ∨ fix_deleted_field fx = true)
  ∧ fix_refine fx = true ∧ fix_null fx = true.

Lemma all_fixes_sound : fixes_sound all_fixes.
Proof.
  split; [ reflexivity | split; [ left; reflexivity | split; reflexivity ] ].
Qed.

Lemma landed_fixes_sound : fixes_sound landed_fixes.
Proof.
  split; [ reflexivity | split; [ right; reflexivity | split; reflexivity ] ].
Qed.

(* The rewriting is sound in the fragment of [loose]. *)
Lemma rewrite_type_sound_in (fx : fixes) (loose : bool) :
  fixes_sound fx ->
  ∀ (is_local : field -> bool) (F : facts) (T : tables)
    (sets : list set_of_closures) (tenv : Typing_env.t) (canon : canon_fn)
    (statics statics' : static_env) (valuation valuation' : env)
    (fuel : nat) (m : rewrite_metadata) (t : flambda_type) (v v' : value),
    least_either is_local F T ->
    is_local_slots_only is_local ->
    sets_of_graph (graph_of_facts F) sets ->
    typing_env_satisfied_strictly statics valuation tenv ->
    env_in_fragment_with loose valuation tenv ->
    statics_allocated (solution_of_tables T) statics ->
    statics_rebuilt (graph_of_facts F) (solution_of_tables T) statics
      statics' ->
    (loose = true ->
     static_sets_whole (solution_of_tables T) statics statics') ->
    copies_valued tenv canon valuation valuation' ->
    type_in_fragment_with loose valuation t ->
    rebuilt_under (graph_of_facts F) (solution_of_tables T) m v v' ->
    type_satisfied statics valuation v t ->
    type_satisfied statics' valuation' v'
      (rewrite_arbitrary fx (plain_context is_local F T sets) tenv canon fuel
         m t).
Proof.
  intros (Hb & Hal & Hr & Hn) is_local F T sets tenv canon statics statics'
    valuation valuation' fuel m t v v' Hleast Hlocal Hsets Henv Hfrag Halloc
    Hst Hwhole Hcopies Ht Hv Hsat.
  exact (proj1 (rewrite_sound fx Hb Hal Hr Hn
                  is_local F T sets tenv canon statics statics' valuation
                  valuation' loose (least_closed_tables is_local F T Hleast)
                  Hleast Hlocal Hsets Henv Hfrag Halloc Hst Hcopies Hwhole
                  fuel)
           m t v v' Ht (or_introl Hv) Hsat).
Qed.

Theorem rewrite_type_sound : rewrite_type_sound_statement.
Proof.
  intros is_local F T sets tenv canon statics statics' valuation valuation'
    fuel m t v v' Hleast Hlocal Hsets Henv Hfrag Halloc Hst Hcopies.
  assert (Hle : least_either is_local F T) by (left; exact Hleast).
  apply (rewrite_type_sound_in all_fixes false all_fixes_sound);
    try assumption.
  intros H. discriminate H.
Qed.

Theorem rewrite_type_sound_local : rewrite_type_sound_local_statement.
Proof.
  intros is_local F T sets tenv canon statics statics' valuation valuation'
    fuel m t v v' Hleast Hlocal Hsets Henv Hfrag Halloc Hst Hcopies.
  assert (Hle : least_either is_local F T) by (right; exact Hleast).
  apply (rewrite_type_sound_in all_fixes false all_fixes_sound);
    try assumption.
  intros H. discriminate H.
Qed.

Theorem rewrite_type_sound_local_landed :
  rewrite_type_sound_local_landed_statement.
Proof.
  intros is_local F T sets tenv canon statics statics' valuation valuation'
    fuel m t v v' Hleast Hlocal Hsets Henv Hfrag Halloc Hst Hcopies.
  assert (Hle : least_either is_local F T) by (right; exact Hleast).
  apply (rewrite_type_sound_in landed_fixes false landed_fixes_sound);
    try assumption.
  intros H. discriminate H.
Qed.

Theorem rewrite_unaliased_type_sound :
  rewrite_unaliased_type_sound_statement.
Proof.
  intros is_local F T sets tenv canon statics statics' valuation valuation'
    fuel m t v v' Hleast Hlocal Hsets Henv Hfrag Halloc Hst Hwhole.
  assert (Hle : least_either is_local F T) by (left; exact Hleast).
  apply (rewrite_type_sound_in all_fixes true all_fixes_sound);
    try assumption.
  intros _. exact Hwhole.
Qed.

Theorem rewrite_unaliased_type_sound_landed :
  rewrite_unaliased_type_sound_landed_statement.
Proof.
  intros is_local F T sets tenv canon statics statics' valuation valuation'
    fuel m t v v' Hleast Hlocal Hsets Henv Hfrag Halloc Hst Hwhole.
  assert (Hle : least_either is_local F T) by (left; exact Hleast).
  apply (rewrite_type_sound_in landed_fixes true landed_fixes_sound);
    try assumption.
  intros _. exact Hwhole.
Qed.

Theorem rewrite_unaliased_type_sound_local :
  rewrite_unaliased_type_sound_local_statement.
Proof.
  intros is_local F T sets tenv canon statics statics' valuation valuation'
    fuel m t v v' Hleast Hlocal Hsets Henv Hfrag Halloc Hst Hwhole.
  assert (Hle : least_either is_local F T) by (right; exact Hleast).
  apply (rewrite_type_sound_in all_fixes true all_fixes_sound);
    try assumption.
  intros _. exact Hwhole.
Qed.

Theorem rewrite_unaliased_type_sound_local_landed :
  rewrite_unaliased_type_sound_local_landed_statement.
Proof.
  intros is_local F T sets tenv canon statics statics' valuation valuation'
    fuel m t v v' Hleast Hlocal Hsets Henv Hfrag Halloc Hst Hwhole.
  assert (Hle : least_either is_local F T) by (right; exact Hleast).
  apply (rewrite_type_sound_in landed_fixes true landed_fixes_sound);
    try assumption.
  intros _. exact Hwhole.
Qed.

(* The type of a symbol in the fragment of [loose]. *)
Lemma rewrite_symbol_type_sound_in (fx : fixes) (loose : bool) :
  fixes_sound fx ->
  ∀ (is_local : field -> bool) (F : facts) (T : tables)
    (sets : list set_of_closures) (tenv : Typing_env.t) (canon : canon_fn)
    (statics statics' : static_env) (valuation valuation' : env)
    (s : symbol) (allocated : statically_allocated_value),
    least_either is_local F T ->
    is_local_slots_only is_local ->
    sets_of_graph (graph_of_facts F) sets ->
    typing_env_satisfied_strictly statics valuation tenv ->
    env_in_fragment_with loose valuation tenv ->
    statics_allocated (solution_of_tables T) statics ->
    statics_rebuilt (graph_of_facts F) (solution_of_tables T) statics
      statics' ->
    (loose = true ->
     static_sets_whole (solution_of_tables T) statics statics') ->
    copies_valued tenv canon valuation valuation' ->
    lookup s (static_values statics) = Some allocated ->
    type_satisfied statics' valuation' (Value_symbol s)
      (rewrite_symbol_type fx (plain_context is_local F T sets) tenv canon s).
Proof.
  intros (Hb & Hal & Hr & Hn) is_local F T sets tenv canon statics statics'
    valuation valuation' s allocated Hleast Hlocal Hsets Henv Hfrag Halloc Hst
    Hwhole Hcopies Hl.
  unfold rewrite_symbol_type, rewrite_concrete_type_of.
  set (ty := concrete_type_of tenv (Name_symbol s) Value).
  assert (Hty : type_in_fragment_with loose valuation ty
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
  apply (proj2 (rewrite_sound fx Hb Hal Hr Hn
                  is_local F T sets tenv canon statics statics' valuation
                  valuation' loose (least_closed_tables is_local F T Hleast)
                  Hleast Hlocal Hsets Henv Hfrag Halloc Hst Hcopies Hwhole
                  (rewrite_fuel tenv ty))
           _ ty (Value_symbol s) (Value_symbol s) Hfr);
    [ | exact Hsat ].
  left. right. exists (Node_symbol s).
  split; [ | split; [ exact Hat | exact Hhu ] ].
  cbn [rebuilt_value].
  split; [ reflexivity | split; [ exact (Halloc s allocated Hl) | auto ] ].
Qed.

Theorem rewrite_symbol_type_sound : rewrite_symbol_type_sound_statement.
Proof.
  intros is_local F T sets tenv canon statics statics' valuation valuation' s
    allocated Hleast Hlocal Hsets Henv Hfrag Halloc Hst.
  assert (Hle : least_either is_local F T) by (left; exact Hleast).
  apply (rewrite_symbol_type_sound_in all_fixes false all_fixes_sound);
    try assumption.
  intros H. discriminate H.
Qed.

Theorem rewrite_symbol_type_sound_local :
  rewrite_symbol_type_sound_local_statement.
Proof.
  intros is_local F T sets tenv canon statics statics' valuation valuation' s
    allocated Hleast Hlocal Hsets Henv Hfrag Halloc Hst.
  assert (Hle : least_either is_local F T) by (right; exact Hleast).
  apply (rewrite_symbol_type_sound_in all_fixes false all_fixes_sound);
    try assumption.
  intros H. discriminate H.
Qed.

Theorem rewrite_symbol_type_sound_local_landed :
  rewrite_symbol_type_sound_local_landed_statement.
Proof.
  intros is_local F T sets tenv canon statics statics' valuation valuation' s
    allocated Hleast Hlocal Hsets Henv Hfrag Halloc Hst.
  assert (Hle : least_either is_local F T) by (right; exact Hleast).
  apply (rewrite_symbol_type_sound_in landed_fixes false landed_fixes_sound);
    try assumption.
  intros H. discriminate H.
Qed.

Theorem rewrite_unaliased_symbol_type_sound :
  rewrite_unaliased_symbol_type_sound_statement.
Proof.
  intros is_local F T sets tenv canon statics statics' valuation valuation' s
    allocated Hleast Hlocal Hsets Henv Hfrag Halloc Hst Hwhole.
  assert (Hle : least_either is_local F T) by (left; exact Hleast).
  apply (rewrite_symbol_type_sound_in all_fixes true all_fixes_sound);
    try assumption.
  intros _. exact Hwhole.
Qed.

Theorem rewrite_unaliased_symbol_type_sound_landed :
  rewrite_unaliased_symbol_type_sound_landed_statement.
Proof.
  intros is_local F T sets tenv canon statics statics' valuation valuation' s
    allocated Hleast Hlocal Hsets Henv Hfrag Halloc Hst Hwhole.
  assert (Hle : least_either is_local F T) by (left; exact Hleast).
  apply (rewrite_symbol_type_sound_in landed_fixes true landed_fixes_sound);
    try assumption.
  intros _. exact Hwhole.
Qed.

Theorem rewrite_unaliased_symbol_type_sound_local :
  rewrite_unaliased_symbol_type_sound_local_statement.
Proof.
  intros is_local F T sets tenv canon statics statics' valuation valuation' s
    allocated Hleast Hlocal Hsets Henv Hfrag Halloc Hst Hwhole.
  assert (Hle : least_either is_local F T) by (right; exact Hleast).
  apply (rewrite_symbol_type_sound_in all_fixes true all_fixes_sound);
    try assumption.
  intros _. exact Hwhole.
Qed.

Theorem rewrite_unaliased_symbol_type_sound_local_landed :
  rewrite_unaliased_symbol_type_sound_local_landed_statement.
Proof.
  intros is_local F T sets tenv canon statics statics' valuation valuation' s
    allocated Hleast Hlocal Hsets Henv Hfrag Halloc Hst Hwhole.
  assert (Hle : least_either is_local F T) by (right; exact Hleast).
  apply (rewrite_symbol_type_sound_in landed_fixes true landed_fixes_sound);
    try assumption.
  intros _. exact Hwhole.
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

(* The type of a result in the fragment of [loose]. *)
Lemma rewrite_result_type_sound_in (fx : fixes) (loose : bool) :
  fixes_sound fx ->
  ∀ (is_local : field -> bool) (F : facts) (T : tables)
    (sets : list set_of_closures) (tenv : Typing_env.t) (canon : canon_fn)
    (statics statics' : static_env) (valuation valuation' : env)
    (code_var type_var : variable) (v v' : value),
    least_either is_local F T ->
    is_local_slots_only is_local ->
    sets_of_graph (graph_of_facts F) sets ->
    typing_env_satisfied_strictly statics valuation tenv ->
    env_in_fragment_with loose valuation tenv ->
    statics_allocated (solution_of_tables T) statics ->
    statics_rebuilt (graph_of_facts F) (solution_of_tables T) statics
      statics' ->
    (loose = true ->
     static_sets_whole (solution_of_tables T) statics statics') ->
    copies_valued tenv canon valuation valuation' ->
    Typing_env.find tenv (Name_var type_var) None <> None ->
    lookup type_var valuation = Some v ->
    rebuilt_value (graph_of_facts F) (solution_of_tables T) v
      (Node_variable code_var) v' ->
    type_satisfied statics' valuation' v'
      (rewrite_result_type fx (plain_context is_local F T sets) tenv canon
         code_var type_var).
Proof.
  intros (Hb & Hal & Hr & Hn) is_local F T sets tenv canon statics statics'
    valuation valuation' code_var type_var v v' Hleast Hlocal Hsets Henv Hfrag
    Halloc Hst Hwhole Hcopies Hfind Hl Hv.
  unfold rewrite_result_type, rewrite_concrete_type_of, kind_of_name.
  destruct (Typing_env.find tenv (Name_var type_var) None) as [ty0 | ] eqn:Hf;
    [ | contradiction ].
  unfold concrete_type_of. rewrite (find_with_kind tenv _ ty0 Hf).
  pose proof (find_fragment valuation tenv Hfrag _ _ ty0 Hf) as Hfr.
  pose proof (find_satisfied_strictly statics valuation tenv (Name_var type_var)
                None ty0 v Henv Hf Hl) as Hsat.
  pose proof (metadata_for_name_at (plain_context is_local F T sets)
                (Node_variable code_var)) as [Hat Hhu].
  apply (proj2 (rewrite_sound fx Hb Hal Hr Hn
                  is_local F T sets tenv canon statics statics' valuation
                  valuation' loose (least_closed_tables is_local F T Hleast)
                  Hleast Hlocal Hsets Henv Hfrag Halloc Hst Hcopies Hwhole
                  (rewrite_fuel tenv ty0))
           _ ty0 v v' Hfr);
    [ | exact Hsat ].
  left. right. exists (Node_variable code_var).
  split; [ exact Hv | split; [ exact Hat | exact Hhu ] ].
Qed.

Theorem rewrite_result_type_sound : rewrite_result_type_sound_statement.
Proof.
  intros is_local F T sets tenv canon statics statics' valuation valuation'
    code_var type_var v v' Hleast Hlocal Hsets Henv Hfrag Halloc Hst.
  assert (Hle : least_either is_local F T) by (left; exact Hleast).
  apply (rewrite_result_type_sound_in all_fixes false all_fixes_sound
           is_local F T sets tenv canon statics statics' valuation
           valuation'); try assumption.
  intros H. discriminate H.
Qed.

Theorem rewrite_result_type_sound_local :
  rewrite_result_type_sound_local_statement.
Proof.
  intros is_local F T sets tenv canon statics statics' valuation valuation'
    code_var type_var v v' Hleast Hlocal Hsets Henv Hfrag Halloc Hst.
  assert (Hle : least_either is_local F T) by (right; exact Hleast).
  apply (rewrite_result_type_sound_in all_fixes false all_fixes_sound
           is_local F T sets tenv canon statics statics' valuation
           valuation'); try assumption.
  intros H. discriminate H.
Qed.

Theorem rewrite_result_type_sound_local_landed :
  rewrite_result_type_sound_local_landed_statement.
Proof.
  intros is_local F T sets tenv canon statics statics' valuation valuation'
    code_var type_var v v' Hleast Hlocal Hsets Henv Hfrag Halloc Hst.
  assert (Hle : least_either is_local F T) by (right; exact Hleast).
  apply (rewrite_result_type_sound_in landed_fixes false landed_fixes_sound
           is_local F T sets tenv canon statics statics' valuation
           valuation'); try assumption.
  intros H. discriminate H.
Qed.

Theorem rewrite_unaliased_result_type_sound :
  rewrite_unaliased_result_type_sound_statement.
Proof.
  intros is_local F T sets tenv canon statics statics' valuation valuation'
    code_var type_var v v' Hleast Hlocal Hsets Henv Hfrag Halloc Hst Hwhole.
  assert (Hle : least_either is_local F T) by (left; exact Hleast).
  apply (rewrite_result_type_sound_in all_fixes true all_fixes_sound
           is_local F T sets tenv canon statics statics' valuation
           valuation'); try assumption.
  intros _. exact Hwhole.
Qed.

Theorem rewrite_unaliased_result_type_sound_landed :
  rewrite_unaliased_result_type_sound_landed_statement.
Proof.
  intros is_local F T sets tenv canon statics statics' valuation valuation'
    code_var type_var v v' Hleast Hlocal Hsets Henv Hfrag Halloc Hst Hwhole.
  assert (Hle : least_either is_local F T) by (left; exact Hleast).
  apply (rewrite_result_type_sound_in landed_fixes true landed_fixes_sound
           is_local F T sets tenv canon statics statics' valuation
           valuation'); try assumption.
  intros _. exact Hwhole.
Qed.

Theorem rewrite_unaliased_result_type_sound_local :
  rewrite_unaliased_result_type_sound_local_statement.
Proof.
  intros is_local F T sets tenv canon statics statics' valuation valuation'
    code_var type_var v v' Hleast Hlocal Hsets Henv Hfrag Halloc Hst Hwhole.
  assert (Hle : least_either is_local F T) by (right; exact Hleast).
  apply (rewrite_result_type_sound_in all_fixes true all_fixes_sound
           is_local F T sets tenv canon statics statics' valuation
           valuation'); try assumption.
  intros _. exact Hwhole.
Qed.

Theorem rewrite_unaliased_result_type_sound_local_landed :
  rewrite_unaliased_result_type_sound_local_landed_statement.
Proof.
  intros is_local F T sets tenv canon statics statics' valuation valuation'
    code_var type_var v v' Hleast Hlocal Hsets Henv Hfrag Halloc Hst Hwhole.
  assert (Hle : least_either is_local F T) by (right; exact Hleast).
  apply (rewrite_result_type_sound_in landed_fixes true landed_fixes_sound
           is_local F T sets tenv canon statics statics' valuation
           valuation'); try assumption.
  intros _. exact Hwhole.
Qed.

(* ======================================================================== *)
(* The two runs                                                             *)
(* ======================================================================== *)

Lemma exported_types_sound_gen (fx : fixes) :
  fixes_sound fx -> exported_types_sound_with fx.
Proof.
  intros Hfx flag is_local u e T preserve_direct_calls sets tenv canon
    valuation valuation' n s k vs hp rd sym allocated Hlocal Hleast Hsets
    Hrel Hrun Hstep Henv Hfrag Hcopies Hl.
  destruct (Hrel n s k vs hp rd Hrun Hstep)
    as [Halloc (m & t & vs' & rd' & Hrun' & Hstep' & Hst)].
  exists m, t, vs', rd'. split; [ exact Hrun' | split; [ exact Hstep' | ] ].
  apply (rewrite_symbol_type_sound_in fx false Hfx is_local
           (traverse_unit flag u e) T sets tenv canon (state_statics s)
           (state_statics t) valuation valuation' sym allocated);
    try assumption.
  - left. exact Hleast.
  - intros H. discriminate H.
Qed.

Theorem exported_types_sound : exported_types_sound_statement.
Proof. exact (exported_types_sound_gen all_fixes all_fixes_sound). Qed.

Theorem exported_types_sound_local : exported_types_sound_local_statement.
Proof.
  intros flag is_local u e T preserve_direct_calls sets tenv canon valuation
    valuation' n s k vs hp rd sym allocated Hlocal Hleast Hsets Hrel Hrun
    Hstep Henv Hfrag Hcopies Hl.
  destruct (Hrel n s k vs hp rd Hrun Hstep)
    as [Halloc (m & t & vs' & rd' & Hrun' & Hstep' & Hst)].
  exists m, t, vs', rd'. split; [ exact Hrun' | split; [ exact Hstep' | ] ].
  exact (rewrite_symbol_type_sound_local is_local (traverse_unit flag u e) T
           sets tenv canon (state_statics s) (state_statics t) valuation
           valuation' sym allocated
           Hleast Hlocal Hsets Henv Hfrag Halloc Hst Hcopies Hl).
Qed.

Theorem exported_types_sound_local_landed :
  exported_types_sound_local_landed_statement.
Proof.
  intros flag is_local u e T preserve_direct_calls sets tenv canon valuation
    valuation' n s k vs hp rd sym allocated Hlocal Hleast Hsets Hrel Hrun
    Hstep Henv Hfrag Hcopies Hl.
  destruct (Hrel n s k vs hp rd Hrun Hstep)
    as [Halloc (m & t & vs' & rd' & Hrun' & Hstep' & Hst)].
  exists m, t, vs', rd'. split; [ exact Hrun' | split; [ exact Hstep' | ] ].
  exact (rewrite_symbol_type_sound_local_landed is_local
           (traverse_unit flag u e) T sets tenv canon (state_statics s)
           (state_statics t) valuation valuation' sym allocated
           Hleast Hlocal Hsets Henv Hfrag Halloc Hst Hcopies Hl).
Qed.

Theorem exported_unaliased_types_sound :
  exported_unaliased_types_sound_statement.
Proof.
  intros flag is_local u e T preserve_direct_calls sets tenv canon valuation
    valuation' n s k vs hp rd sym allocated Hlocal Hleast Hsets Hrel Hrun
    Hstep Henv Hfrag Hcopies Hl.
  destruct (Hrel n s k vs hp rd Hrun Hstep)
    as [Halloc (m & t & vs' & rd' & Hrun' & Hstep' & Hst & Hwhole)].
  exists m, t, vs', rd'. split; [ exact Hrun' | split; [ exact Hstep' | ] ].
  exact (rewrite_unaliased_symbol_type_sound is_local (traverse_unit flag u e)
           T sets tenv canon (state_statics s) (state_statics t) valuation
           valuation' sym allocated Hleast Hlocal Hsets Henv Hfrag Halloc Hst
           Hwhole Hcopies Hl).
Qed.

Theorem exported_unaliased_types_sound_landed :
  exported_unaliased_types_sound_landed_statement.
Proof.
  intros flag is_local u e T preserve_direct_calls sets tenv canon valuation
    valuation' n s k vs hp rd sym allocated Hlocal Hleast Hsets Hrel Hrun
    Hstep Henv Hfrag Hcopies Hl.
  destruct (Hrel n s k vs hp rd Hrun Hstep)
    as [Halloc (m & t & vs' & rd' & Hrun' & Hstep' & Hst & Hwhole)].
  exists m, t, vs', rd'. split; [ exact Hrun' | split; [ exact Hstep' | ] ].
  exact (rewrite_unaliased_symbol_type_sound_landed is_local
           (traverse_unit flag u e) T sets tenv canon (state_statics s)
           (state_statics t) valuation valuation' sym allocated Hleast Hlocal
           Hsets Henv Hfrag Halloc Hst Hwhole Hcopies Hl).
Qed.

Theorem exported_unaliased_types_sound_local :
  exported_unaliased_types_sound_local_statement.
Proof.
  intros flag is_local u e T preserve_direct_calls sets tenv canon valuation
    valuation' n s k vs hp rd sym allocated Hlocal Hleast Hsets Hrel Hrun
    Hstep Henv Hfrag Hcopies Hl.
  destruct (Hrel n s k vs hp rd Hrun Hstep)
    as [Halloc (m & t & vs' & rd' & Hrun' & Hstep' & Hst & Hwhole)].
  exists m, t, vs', rd'. split; [ exact Hrun' | split; [ exact Hstep' | ] ].
  exact (rewrite_unaliased_symbol_type_sound_local is_local
           (traverse_unit flag u e) T sets tenv canon (state_statics s)
           (state_statics t) valuation valuation' sym allocated Hleast Hlocal
           Hsets Henv Hfrag Halloc Hst Hwhole Hcopies Hl).
Qed.

Theorem exported_unaliased_types_sound_local_landed :
  exported_unaliased_types_sound_local_landed_statement.
Proof.
  intros flag is_local u e T preserve_direct_calls sets tenv canon valuation
    valuation' n s k vs hp rd sym allocated Hlocal Hleast Hsets Hrel Hrun
    Hstep Henv Hfrag Hcopies Hl.
  destruct (Hrel n s k vs hp rd Hrun Hstep)
    as [Halloc (m & t & vs' & rd' & Hrun' & Hstep' & Hst & Hwhole)].
  exists m, t, vs', rd'. split; [ exact Hrun' | split; [ exact Hstep' | ] ].
  exact (rewrite_unaliased_symbol_type_sound_local_landed is_local
           (traverse_unit flag u e) T sets tenv canon (state_statics s)
           (state_statics t) valuation valuation' sym allocated Hleast Hlocal
           Hsets Henv Hfrag Halloc Hst Hwhole Hcopies Hl).
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
  - destruct fx as [b1 b2 b3 b4 b5]. cbn in Hfx. subst b1.
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

Theorem deleted_field_example : deleted_field_example_statement.
Proof. split; vm_compute; reflexivity. Qed.

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

Theorem refine_example_landed : refine_example_landed_statement.
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

Theorem example_landed : example_landed_statement.
Proof. split; vm_compute; reflexivity. Qed.

Theorem result_types_example : result_types_example_statement.
Proof. split; vm_compute; reflexivity. Qed.

Theorem result_types_example_landed : result_types_example_landed_statement.
Proof. split; vm_compute; reflexivity. Qed.

(* ======================================================================== *)
(* Unboxed parameters and results                                           *)
(* ======================================================================== *)

Module U := FlambdaReaperUnboxing.
Module URP := FlambdaReaperUnboxingRefinesProofs.
Module UCS := FlambdaReaperUnboxingRefinesCallsSpec.

Section Unboxing_facts.

Variable F : facts.
Variable T : tables.
Variable cm : finite_map variable.
Variable dominated : list (node * node).
Variable to_unbox : list node.

(* A name that an allocation to unbox dominates has unboxed fields. *)
Lemma unboxed_fields_of_names_has (next : positive) (y : node) :
  U.has_to_be_unboxed dominated to_unbox y = true ->
  ∃ fs : U.unboxed_fields,
    (y, fs) ∈ fst (U.unboxed_fields_of_names F T cm dominated to_unbox next).
Proof.
  unfold U.unboxed_fields_of_names, U.has_to_be_unboxed.
  intros Hy. apply existsb_exists in Hy as ([a y'] & Hin & Hy).
  apply andb_true_iff in Hy as [Hyy Ha]. apply node_eqb_eq in Hyy. subst y'.
  assert (Hgen : ∀ (l : list (node * node))
                   (acc : list (node * U.unboxed_fields)) (n : positive),
    (∃ fs, (y, fs) ∈ acc)
    ∨ (∃ a, (a, y) ∈ l ∧ mem_node a to_unbox = true) ->
    ∃ fs,
      (y, fs) ∈ fst (List.fold_left
                       (fun '(acc, next) '(a, x) =>
                          if mem_node a to_unbox
                             && negb (existsb (fun '(y, _) => node_eqb x y)
                                        acc)
                          then
                            let '(fields, next) :=
                              U.name_unboxed_fields F T cm dominated to_unbox
                                a x next
                            in
                            (acc ++ [(x, fields)], next)
                          else (acc, next))
                       l (acc, n))).
  { induction l as [ | [a0 x0] l IH]; intros acc n Hacc.
    - destruct Hacc as [Hacc | (a' & [] & _)]. exact Hacc.
    - cbn [List.fold_left].
      destruct (mem_node a0 to_unbox
                && negb (existsb (fun '(y, _) => node_eqb x0 y) acc))
        eqn:Hc.
      + destruct (U.name_unboxed_fields F T cm dominated to_unbox a0 x0 n)
          as [fields n'].
        apply IH.
        destruct Hacc as [(fs & Hfs) | (a' & [Heq | Hl] & Ha')].
        * left. exists fs. apply in_or_app. left. exact Hfs.
        * injection Heq as -> ->. left. exists fields. apply in_or_app.
          right. left. reflexivity.
        * right. exists a'. auto.
      + apply IH.
        destruct Hacc as [(fs & Hfs) | (a' & [Heq | Hl] & Ha')].
        * left. exists fs. exact Hfs.
        * injection Heq as -> ->. rewrite Ha' in Hc.
          cbn [andb] in Hc. apply negb_false_iff in Hc.
          apply existsb_exists in Hc
            as ([y' fs] & Hfs & Hyy). apply node_eqb_eq in Hyy. subst y'.
          left. exists fs. exact Hfs.
        * right. exists a'. auto. }
  apply (Hgen dominated [] next). right. exists a. auto.
Qed.

(* A field unboxed in turn: read into names all to unbox, and unboxed as
   the fields of its source read from their uses. *)
Lemma mk_unboxed_fields_nested (fuel : nat) (block : node) :
  ∀ (fields : list (field * U.field_use)) (next : positive) (f : field)
    (nested : U.unboxed_fields),
    (f, U.Unboxed nested)
      ∈ fst (U.mk_unboxed_fields F T cm dominated to_unbox fuel block fields
               next) ->
    ∃ (flow_to : list node) (fuel' : nat) (source : node) (next' : positive),
      (f, U.Use_known flow_to) ∈ fields
      ∧ forallb (U.has_to_be_unboxed dominated to_unbox) flow_to = true
      ∧ nested
        = fst (U.mk_unboxed_fields F T cm dominated to_unbox fuel' source
                 (U.get_fields F T
                    (U.add_usages_through_function_slots F T cm true
                       (U.get_direct_usages T flow_to)))
                 next').
Proof.
  destruct fuel as [ | fuel]; cbn [U.mk_unboxed_fields];
    (induction fields as [ | [f' use] fields IH]; intros next f nested Hin;
     [ destruct Hin | ]);
    cbn beta iota zeta in Hin;
    repeat match type of Hin with
    | context [match ?c with _ => _ end] =>
      lazymatch c with
      | (_, _) => fail
      | _ => destruct c eqn:?
      end
    end;
    simpl in Hin.
  all: try (destruct Hin as [Heq | Hin]; [ discriminate Heq | ]).
  all: try (match goal with
            | H : ?g = (?rest, _), Hin : _ ∈ ?rest |- _ =>
              assert (Hin' : (f, U.Unboxed nested) ∈ fst g)
                by (rewrite H; exact Hin);
              destruct (IH _ f nested Hin')
                as (flow_to' & fuel' & s' & n' & H1 & H2 & H3);
              exists flow_to', fuel', s', n';
              split; [ right; exact H1 | split; assumption ]
            end).
  all: repeat match goal with
         | H : (if ?c then _ else _) = (_, _) |- _ => destruct c eqn:?
         | H : (match ?c with _ => _ end) = (_, _) |- _ =>
           lazymatch c with
           | (_, _) => fail
           | _ => destruct c eqn:?
           end
         end.
  all: repeat match goal with
         | H : (_, _) = (_, _) |- _ => injection H as; subst
         | H : None = Some _ |- _ => discriminate H
         end.
  all: try (destruct Hin as [Heq | Hin]; [ discriminate Heq | ]).
  all: try (match goal with
            | H : ?g = (?rest, _), Hin : _ ∈ ?rest |- _ =>
              assert (Hin' : (f, U.Unboxed nested) ∈ fst g)
                by (rewrite H; exact Hin);
              destruct (IH _ f nested Hin')
                as (flow_to' & fuel' & s' & n' & H1 & H2 & H3);
              exists flow_to', fuel', s', n';
              split; [ right; exact H1 | split; assumption ]
            end).
  all: destruct Hin as [Heq | Hin];
    [ injection Heq as -> ->;
      match goal with
      | H : U.mk_unboxed_fields _ _ _ _ _ ?fu ?so _ ?ne = (_, _),
        Hf : forallb _ ?fl = true |- _ =>
        exists fl, fu, so, ne;
        split; [ left; reflexivity
               | split; [ exact Hf | rewrite H; reflexivity ] ]
      end
    | ].
  all: match goal with
       | H : ?g = (?rest, _), Hin : _ ∈ ?rest |- _ =>
         assert (Hin' : (f, U.Unboxed nested) ∈ fst g)
           by (rewrite H; exact Hin);
         destruct (IH _ f nested Hin')
           as (flow_to' & fuel' & s' & n' & H1 & H2 & H3);
         exists flow_to', fuel', s', n';
         split; [ right; exact H1 | split; assumption ]
       end.
Qed.

(* The uses of the names of [unboxed]: with [check_unboxed_uses], the
   closures among them add nothing. *)
Lemma add_usages_of_unboxed (unboxed : list (node * U.unboxed_fields))
    (follow : bool) (out : list node) :
  UCS.check_unboxed_uses F T unboxed = true ->
  (∀ u : node,
     u ∈ out ->
     ∃ (x : node) (fs : U.unboxed_fields),
       (x, fs) ∈ unboxed ∧ mem_pair x u (t_usages T) = true) ->
  U.add_usages_through_function_slots F T cm follow out = out.
Proof.
  intros Huses Hout.
  assert (Hblock : ∀ x, x ∈ out ->
            ∀ y f base, (y, f, base) ∈ f_accessor F ->
              node_eqb base x = true -> U.is_block_like_field f = true).
  { intros x Hx y f base Hacc Hb. apply node_eqb_eq in Hb. subst base.
    destruct (Hout x Hx) as (n & fs & Hn & Hpair).
    unfold UCS.check_unboxed_uses in Huses. rewrite forallb_forall in Huses.
    specialize (Huses _ Hacc). cbn beta iota in Huses.
    apply orb_true_iff in Huses as [Hf | Hall]; [ exact Hf | ].
    rewrite forallb_forall in Hall. specialize (Hall _ Hn).
    cbn beta iota in Hall. rewrite Hpair in Hall. discriminate Hall. }
  clear Hout.
  assert (Hnone : U.usages_through_function_slots F T cm follow out = []).
  { destruct (U.usages_through_function_slots F T cm follow out)
      as [ | z rest] eqn:E; [ reflexivity | exfalso ].
    assert (Hz : z ∈ U.usages_through_function_slots F T cm follow out)
      by (rewrite E; left; reflexivity).
    clear E. unfold U.usages_through_function_slots in Hz.
    apply in_flat_map in Hz as (x & Hx & Hz).
    apply in_app_or in Hz as [Hz | Hz];
      [ destruct follow; [ | destruct Hz ] | ];
      apply in_flat_map in Hz as ([[y f] base] & Hacc & Hz);
      cbn beta iota in Hz;
      match type of Hz with
      | _ ∈ (if ?c then _ else _) => destruct c eqn:Ht; [ | destruct Hz ]
      end;
      apply andb_true_iff in Ht as [Hb Hf];
      pose proof (Hblock x Hx y f base Hacc Hb) as Hbl;
      destruct f; try discriminate Hbl; discriminate Hf. }
  unfold U.add_usages_through_function_slots. simpl. rewrite Hnone.
  reflexivity.
Qed.

(* [get_fields] reads the nodes given as a set. *)
Lemma get_fields_ext (uses uses' : list node) :
  (∀ z : node, mem_node z uses = mem_node z uses') ->
  U.get_fields F T uses = U.get_fields F T uses'.
Proof.
  intros H. unfold U.get_fields.
  match goal with
  | |- U.sort_fields (flat_map _ (Keyed.dedup _ (map _ (filter ?p ?l))))
       = U.sort_fields (flat_map _ (Keyed.dedup _ (map _ (filter ?q ?l)))) =>
    replace (filter p l) with (filter q l); [ reflexivity | ]
  end.
  apply filter_ext. intros [[y f] base]. rewrite H. reflexivity.
Qed.

(* A field of [get_fields] is there once, with one use. *)
Lemma get_fields_functional (uses : list node) (f : field)
    (use use' : U.field_use) :
  (f, use) ∈ U.get_fields F T uses -> (f, use') ∈ U.get_fields F T uses ->
  use = use'.
Proof.
  unfold U.get_fields. cbv zeta. intros H1 H2.
  apply URP.sort_fields_in, in_flat_map in H1 as (f1 & _ & H1).
  apply URP.sort_fields_in, in_flat_map in H2 as (f2 & _ & H2).
  repeat match goal with
         | H : _ ∈ (if ?c then _ else _) |- _ => destruct c eqn:?
         | H : _ ∈ (match ?c with _ => _ end) |- _ => destruct c eqn:?
         end;
    repeat match goal with
           | H : _ ∈ [] |- _ => destruct H
           | H : _ ∈ [_] |- _ =>
             destruct H as [H | []]; injection H as ? ?; subst
           end;
    try reflexivity; try congruence.
Qed.

(* The use of a field of [get_fields]: of a reader that has any usage, or
   the readers with a use, of which none has any usage. *)
Lemma get_fields_use (uses : list node) (f : field) (use : U.field_use) :
  (f, use) ∈ U.get_fields F T uses ->
  match use with
  | U.Use_unknown =>
    ∃ y u : node,
      g_accessor (graph_of_facts F) y f u ∧ mem_node u uses = true
      ∧ s_any_usage (solution_of_tables T) y = true
  | U.Use_known flow_to =>
    flow_to <> [] ∧ is_function_slot f = false
    ∧ (∀ y u : node,
         g_accessor (graph_of_facts F) y f u -> mem_node u uses = true ->
         is_function_slot f = false ->
         s_any_usage (solution_of_tables T) y = false)
    ∧ ∀ y : node,
        y ∈ flow_to ->
        s_has_usage (solution_of_tables T) y = true
        ∧ ∃ u : node,
            g_accessor (graph_of_facts F) y f u ∧ mem_node u uses = true
  end.
Proof.
  unfold U.get_fields. intros Hin.
  apply URP.sort_fields_in, in_flat_map in Hin as (f0 & _ & Hin).
  match type of Hin with
  | _ ∈ (if existsb _ ?ri then _ else _) =>
    remember ri as read_into eqn:Hri
  end.
  assert (Hri_iff : ∀ y : node,
            y ∈ read_into
            <-> ∃ base : node,
                  (y, f0, base) ∈ f_accessor F ∧ mem_node base uses = true
                  ∧ is_function_slot f0 = false).
  { intros y. subst read_into. rewrite in_flat_map. split.
    - intros ([[v f'] base] & Hr & Hv). apply filter_In in Hr as [Hr Ht].
      cbn beta iota in Hv, Ht.
      destruct (field_eqb f0 f') eqn:Hf; [ | destruct Hv ].
      destruct Hv as [<- | []]. apply field_eqb_eq in Hf. subst f'.
      apply andb_true_iff in Ht as [Hb Hfs]. exists base.
      split; [ exact Hr | split; [ exact Hb | ] ].
      apply negb_true_iff. exact Hfs.
    - intros (base & Hr & Hb & Hfs). exists (y, f0, base). split.
      + apply filter_In. split; [ exact Hr | ]. cbn beta iota.
        rewrite Hb, Hfs. reflexivity.
      + cbn beta iota. rewrite field_eqb_refl. left. reflexivity. }
  clear Hri.
  destruct (existsb (fun v => mem_node v (t_any_usage T)) read_into)
    eqn:Hany.
  - destruct Hin as [Heq | []]. injection Heq as -> <-.
    apply existsb_exists in Hany as (y & Hy & Hau).
    destruct (proj1 (Hri_iff y) Hy) as (base & Hr & Hb & _).
    exists y, base. split; [ exact Hr | split; [ exact Hb | exact Hau ] ].
  - destruct (Keyed.dedup node_eqb
                (filter (fun v => mem_node v (t_has_usage T)) read_into))
      as [ | y0 l] eqn:Hdedup; [ destruct Hin | ].
    destruct Hin as [Heq | []]. injection Heq as -> <-.
    split; [ discriminate | split; [ | split ] ].
    + assert (Hy0 : y0 ∈ Keyed.dedup node_eqb
                      (filter (fun v => mem_node v (t_has_usage T))
                         read_into))
        by (rewrite Hdedup; left; reflexivity).
      apply URP.dedup_in, filter_In in Hy0 as [Hy0 _].
      destruct (proj1 (Hri_iff y0) Hy0) as (base & _ & _ & Hfs).
      exact Hfs.
    + intros y u Hacc Hu Hfs. apply not_true_iff_false. intros Hau.
      apply not_true_iff_false in Hany. apply Hany. apply existsb_exists.
      exists y. split; [ | exact Hau ]. apply Hri_iff.
      exists u. split; [ exact Hacc | split; [ exact Hu | exact Hfs ] ].
    + intros y Hy. rewrite <- Hdedup in Hy. apply URP.dedup_in in Hy.
      apply filter_In in Hy as [Hy Hhu].
      destruct (proj1 (Hri_iff y) Hy) as (base & Hr & Hb & _).
      split; [ exact Hhu | ]. exists base. split; [ exact Hr | exact Hb ].
Qed.

End Unboxing_facts.

(* The queries of Types.v and of FlambdaReaperUnboxing.v agree. *)
Lemma direct_usages_agree (ctx : rewrite_context) (xs : list node) (z : node) :
  z ∈ get_direct_usages ctx xs
  <-> z ∈ U.get_direct_usages (rc_tables ctx) xs.
Proof.
  rewrite get_direct_usages_iff. unfold U.get_direct_usages.
  split.
  - intros (x & Hx & Hxz & Hhz). apply URP.dedup_in_r, in_flat_map.
    exists (x, z). cbn [solution_of_tables s_usages s_has_usage] in Hxz, Hhz.
    split; [ apply mem_pair_In; exact Hxz | ].
    apply mem_node_In in Hx. rewrite Hx, Hhz. left. reflexivity.
  - intros Hz. apply URP.dedup_in, in_flat_map in Hz as ([x y] & Hin & Hz).
    destruct (mem_node x xs && mem_node y (t_has_usage (rc_tables ctx)))
      eqn:Ht; [ | destruct Hz ].
    destruct Hz as [<- | []]. apply andb_true_iff in Ht as [Hx Hy].
    exists x. split; [ apply mem_node_In; exact Hx | ].
    cbn [solution_of_tables s_usages s_has_usage].
    split; [ apply mem_pair_In; exact Hin | exact Hy ].
Qed.

Lemma mem_node_iff_eq (l l' : list node) :
  (∀ z : node, z ∈ l <-> z ∈ l') -> ∀ z : node, mem_node z l = mem_node z l'.
Proof.
  intros H z. destruct (mem_node z l) eqn:H1; destruct (mem_node z l') eqn:H2;
    try reflexivity; apply mem_node_In in H1 || apply mem_node_In in H2.
  - apply H, mem_node_In in H1. congruence.
  - apply H, mem_node_In in H2. congruence.
Qed.

(* A field with one source has a node at that field. *)
Lemma get_single_field_source_one_constructor (ctx : rewrite_context)
    (a : node) (f : field) (s : node) :
  get_single_field_source ctx a f = Field_one_source s ->
  ∃ from : node, g_constructor (graph_of_facts (rc_facts ctx)) a f from.
Proof.
  unfold get_single_field_source.
  destruct (existsb (any_source ctx) (constructors_of ctx a f));
    [ discriminate | ].
  destruct (dedup_nodes
              (flat_map (fun from => filter (has_source ctx)
                                       (sources_of ctx from))
                 (constructors_of ctx a f)))
    as [ | s0 [ | s1 rest]] eqn:Hl; try discriminate.
  intros _. assert (Hs0 : s0 ∈ dedup_nodes
                            (flat_map (fun from => filter (has_source ctx)
                                                     (sources_of ctx from))
                               (constructors_of ctx a f)))
    by (rewrite Hl; left; reflexivity).
  apply In_dedup_nodes, in_flat_map in Hs0 as (from & Hfrom & _).
  exists from. apply constructors_of_iff. exact Hfrom.
Qed.

(* The fields of two blocks related position by position. *)
Lemma fields_rebuilt_nth (P : nat -> value -> value -> Prop) :
  ∀ (fields fields' : list value) (k i : nat) (x' : value),
    fields_rebuilt P k fields fields' -> nth_error fields' i = Some x' ->
    ∃ x : value, nth_error fields i = Some x ∧ P (k + i) x x'.
Proof.
  induction fields as [ | x fields IH]; intros [ | y fields'] k i x' H Hn;
    cbn [fields_rebuilt] in H; try contradiction.
  - destruct i; discriminate Hn.
  - destruct H as [Hxy H]. destruct i as [ | i]; cbn in Hn.
    + injection Hn as <-. exists x. split; [ reflexivity | ].
      rewrite Nat.add_0_r. exact Hxy.
    + destruct (IH fields' (S k) i x' H Hn) as (z & Hz & Hp).
      exists z. split; [ exact Hz | ]. rewrite <- Nat.add_succ_comm.
      exact Hp.
Qed.

Section Rebuilt_inv.

Variable G : graph.
Variable sol : solution.

(* What the first run has where the second has a block. *)
Lemma rebuilt_block_inv (w : value) (n : node) (tag : Z)
    (fields' : list value) :
  rebuilt_value G sol w n (Value_block tag fields') ->
  ∃ (fields : list value) (a : node),
    w = Value_block tag fields ∧ may_come_from sol a n
    ∧ s_has_source sol a = true
    ∧ fields_rebuilt (field_rebuilt G sol (rebuilt_value G sol) a) 0 fields
        fields'.
Proof.
  destruct w as [c | code vs | slot decls vs | s | tag0 fields0 | elements
                | c | l | callee args arity];
    cbn [rebuilt_value]; intros H.
  - destruct H as [_ H]. discriminate H.
  - contradiction.
  - destruct H as (sites & vs' & H & _). discriminate H.
  - destruct H as [H _]. discriminate H.
  - destruct H as (a & fields1 & Hm & Hh & Heq & Hf).
    injection Heq as -> ->. exists fields0, a. auto.
  - destruct H as (elements' & H). discriminate H.
  - destruct H as (a & bn & from & _ & _ & _ & _ & H).
    destruct (s_field_used sol a (Field_boxed_number bn)); discriminate H.
  - contradiction.
  - destruct H as (closure' & args' & H). discriminate H.
Qed.

(* What the first run has where the second has a boxed number. *)
Lemma rebuilt_boxed_inv (w : value) (n : node) (c : const) :
  rebuilt_value G sol w n (Value_boxed_number c) -> w = Value_boxed_number c.
Proof.
  destruct w as [c0 | code vs | slot decls vs | s | tag0 fields0 | elements
                | c0 | l | callee args arity];
    cbn [rebuilt_value]; intros H.
  - destruct H as [_ H]. discriminate H.
  - contradiction.
  - destruct H as (sites & vs' & H & _). discriminate H.
  - destruct H as [H _]. discriminate H.
  - destruct H as (a & fields1 & _ & _ & H & _). discriminate H.
  - destruct H as (elements' & H). discriminate H.
  - destruct H as (a & bn & from & _ & _ & _ & _ & H).
    destruct (s_field_used sol a (Field_boxed_number bn));
      [ symmetry; exact H | discriminate H ].
  - contradiction.
  - destruct H as (closure' & args' & H). discriminate H.
Qed.

End Rebuilt_inv.

(* The paths of the leaves of an unboxing are not empty below its root. *)
Lemma unboxed_field_paths_cons (kind : flambda_kind)
    (fields : U.unboxed_fields) (p : list field) (k : flambda_kind) :
  (p, k) ∈ unboxed_field_paths kind (U.Unboxed fields) -> p <> [].
Proof.
  cbn [unboxed_field_paths].
  induction fields as [ | [f u] fields IH]; intros Hin; [ destruct Hin | ].
  apply in_app_or in Hin as [Hin | Hin]; [ | exact (IH Hin) ].
  apply in_map_iff in Hin as (pk & Heq & _). injection Heq as <- _.
  discriminate.
Qed.

(* The leaves of the pattern are those of the unboxing, in order. *)
Lemma unboxed_leaves_paths (ctx : rewrite_context)
    (cm : finite_map variable) :
  ∀ (u : U.unboxed_field) (kind : flambda_kind) (uses : list node)
    (block : node),
    map (fun l => let '(p, k, _) := l in (p, k))
      (unboxed_leaves ctx cm kind uses block u)
    = unboxed_field_paths kind u.
Proof.
  refine (URP.unboxed_field_rec _ _ _).
  - intros v kind uses block. reflexivity.
  - intros fields IH kind uses block. cbn [unboxed_leaves unboxed_field_paths].
    set (fg := forgets_class _). clearbody fg.
    set (reads := U.get_fields _ _ _). clearbody reads.
    revert IH. induction fields as [ | [f u] fields IHl]; intros IH;
      [ reflexivity | ].
    cbn beta iota. rewrite map_app.
    f_equal; [ | exact (IHl (fun f' u' H => IH f' u' (or_intror H))) ].
    assert (Hforgot : map (fun l => let '(p, k, _) := l in (p, k))
                        (forgotten_leaves f u)
                      = map (fun pk => (f :: fst pk, snd pk))
                          (unboxed_field_paths
                             (FlambdaReaperUnboxing.field_kind f) u)).
    { unfold forgotten_leaves. rewrite map_map. reflexivity. }
    destruct (Keyed.lookup field_eqb f reads) as [use | ]; [ | exact Hforgot ].
    destruct fg; [ exact Hforgot | ].
    destruct u as [v | nested]; [ reflexivity | ].
    destruct use as [ | flow_to]; [ exact Hforgot | ].
    destruct (get_single_field_source ctx block f) as [ | s | ];
      try exact Hforgot.
    rewrite map_map.
    rewrite <- (IH f (U.Unboxed nested) (or_introl eq_refl)
                  (FlambdaReaperUnboxing.field_kind f)
                  (get_direct_usages ctx flow_to) s).
    rewrite map_map. apply map_ext. intros [[p k] m]. reflexivity.
Qed.

(* ======================================================================== *)
(* The types of the leaves                                                  *)
(* ======================================================================== *)

Section Destructure.

Context {loose : bool}.
Variable statics : static_env.
Variable valuation : env.
Variable tenv : Typing_env.t.
Hypothesis Henv : typing_env_satisfied_strictly statics valuation tenv.
Hypothesis Hfrag : env_in_fragment_with loose valuation tenv.

(* The type of a field of a block or of a boxed number, destructured from
   an expanded type of the value. *)
Lemma destructure_field_sound (f : field) (e0 : Expanded_type.t)
    (w w1 : value) :
  type_satisfied statics valuation w (Expanded_type.to_type e0) ->
  type_in_fragment_with loose valuation (Expanded_type.to_type e0) ->
  field_of_value f w = Some w1 ->
  (∀ bn : boxable_number,
     f = Field_boxed_number bn -> kind_of_value w1 = unboxed_kind bn) ->
  type_in_fragment_with loose valuation (destructure_field f (Some e0))
  ∧ (type_satisfied statics valuation w1 (destructure_field f (Some e0))
     ∨ destructure_field f (Some e0) = unknown (U.field_kind f)).
Proof.
  intros Hsat Hfr Hfv Hbk.
  assert (Hunk : type_in_fragment_with loose valuation
                   (unknown (U.field_kind f))
                 ∧ (type_satisfied statics valuation w1
                      (unknown (U.field_kind f))
                    ∨ unknown (U.field_kind f) = unknown (U.field_kind f)))
    by (split; [ apply unknown_fragment | right; reflexivity ]).
  destruct f as [i k | ws | g | ep | | | bn | r | ];
    cbn [field_of_value] in Hfv; try discriminate Hfv.
  - (* a field of a block *)
    destruct w as [ | | | | tag fields | | | | ]; try discriminate Hfv.
    destruct e0 as [kd [ | h | ]];
      cbn [destructure_field Expanded_type.descr Expanded_type.to_type
        Expanded_type.kind] in *;
      [ exact Hunk | | exact (False_ind _ (bottom_satisfied _ _ _ _ Hsat)) ].
    destruct h as [hv | hi | kn ns | hr | hreg]; try exact Hunk.
    destruct hv as [non_null is_null].
    rewrite type_satisfied_value in Hsat. cbn [descr_satisfied] in Hsat.
    rewrite head_of_kind_value_satisfied_eq in Hsat.
    destruct Hsat as [[Hnull _] | [_ [Hsat _]]]; [ discriminate Hnull | ].
    destruct non_null as [ | head | ]; [ exact Hunk | | contradiction ].
    pose proof (head_fragment valuation head is_null Hfr) as Hhfr.
    destruct head as [is_int imms get_tag blocks exts uniq | am | kb ct am
                     | closures am | strs];
      cbn [destructure_non_null unknown_accessor bottom_accessor].
    + rewrite variant_satisfied in Hsat.
      destruct Hsat as [(i0 & Hi & _) | (tag' & fields' & Hblock & _ & _ & Hb
                                          & _)];
        [ discriminate Hi | ].
      cbn [block_of] in Hblock. injection Hblock as <- <-.
      destruct blocks as [row | ]; [ | exact Hunk ].
      pose proof (blocks_fragment valuation is_int imms get_tag row exts uniq
                    Hhfr) as [Hcf Hal].
      destruct row as [known other am].
      rewrite row_like_for_blocks_find_iff in Hb.
      unfold destructure_block_field.
      assert (Hcase : ∀ case : row_like_block_case,
                (∀ ty, ty ∈ case_maps_to case ->
                   type_in_fragment_with loose valuation ty)
                -> row_like_block_case_satisfied statics valuation tag fields
                     case ->
                type_in_fragment_with loose valuation
                  match nth_error (case_maps_to case) i with
                  | Some ty => ty
                  | None => unknown k
                  end
                ∧ (type_satisfied statics valuation w1
                     match nth_error (case_maps_to case) i with
                     | Some ty => ty
                     | None => unknown k
                     end
                   ∨ match nth_error (case_maps_to case) i with
                     | Some ty => ty
                     | None => unknown k
                     end = unknown k)).
      { intros case Hcfr (_ & Hfs & _).
        destruct (nth_error (case_maps_to case) i) as [ty | ] eqn:Hn.
        - split; [ exact (Hcfr ty (nth_error_In _ _ Hn)) | left ].
          exact (proj1 (fields_satisfied_nth _ _ _) Hfs i ty w1 Hn Hfv).
        - split; [ apply unknown_fragment | right; reflexivity ]. }
      destruct other as [case | ].
      * destruct known as [ | p known'].
        -- exact (Hcase case (other_case_fragment valuation [] case am Hcf Hal)
                    Hb).
        -- exact Hunk.
      * destruct known as [ | [t [case | ]] [ | p known']]; try exact Hunk.
        cbn [Keyed.lookup] in Hb.
        destruct (Z.eqb tag t) eqn:Ht; [ | contradiction ].
        exact (Hcase case (known_case_fragment valuation _ _ am t case Hcf Hal
                             (or_introl eq_refl))
                 Hb).
    + exact Hunk.
    + rewrite boxed_number_satisfied in Hsat.
      destruct Hsat as (c & Hc & _). discriminate Hc.
    + rewrite closures_satisfied in Hsat.
      destruct Hsat as (slot & fs & vs & Hcl & _). discriminate Hcl.
    + rewrite string_satisfied in Hsat.
      destruct Hsat as (bytes & Hs & _). discriminate Hs.
  - (* the contents of a boxed number *)
    destruct w as [ | | | | | | c | | ]; try discriminate Hfv.
    injection Hfv as <-. specialize (Hbk bn eq_refl). cbn in Hbk.
    destruct e0 as [kd [ | h | ]];
      cbn [destructure_field Expanded_type.descr Expanded_type.to_type
        Expanded_type.kind] in *;
      [ exact Hunk | | exact (False_ind _ (bottom_satisfied _ _ _ _ Hsat)) ].
    destruct h as [hv | hi | kn ns | hr | hreg]; try exact Hunk.
    destruct hv as [non_null is_null].
    rewrite type_satisfied_value in Hsat. cbn [descr_satisfied] in Hsat.
    rewrite head_of_kind_value_satisfied_eq in Hsat.
    destruct Hsat as [[Hnull _] | [_ [Hsat _]]]; [ discriminate Hnull | ].
    destruct non_null as [ | head | ]; [ exact Hunk | | contradiction ].
    pose proof (head_fragment valuation head is_null Hfr) as Hhfr.
    destruct head as [is_int imms get_tag blocks exts uniq | am | kb ct am
                     | closures am | strs];
      cbn [destructure_non_null unknown_accessor bottom_accessor].
    + rewrite variant_satisfied in Hsat.
      destruct Hsat as [(i0 & Hi & _) | (tag' & fields' & Hblock & _)];
        [ discriminate Hi | discriminate Hblock ].
    + rewrite mutable_block_head_satisfied in Hsat. contradiction.
    + rewrite boxed_number_satisfied in Hsat.
      destruct Hsat as (c' & Hc & Hkc & Hct).
      cbn [boxed_number_of] in Hc.
      unfold boxable_contents in Hc.
      destruct (boxable_number_of_const c); [ | discriminate Hc ].
      injection Hc as <-.
      destruct (kind_eqb (unboxed_kind bn) (unboxed_kind kb)) eqn:Hkk.
      * split; [ exact (contents_fragment valuation kb ct am Hhfr) | left ].
        exact Hct.
      * exfalso. rewrite <- Hbk, Hkc, kind_eqb_refl in Hkk. discriminate Hkk.
    + rewrite closures_satisfied in Hsat.
      destruct Hsat as (slot & fs & vs & Hcl & _). discriminate Hcl.
    + rewrite string_satisfied in Hsat.
      destruct Hsat as (bytes & Hs & _). discriminate Hs.
Qed.

End Destructure.

Section Unboxed_sound.

Variable fx : fixes.
Hypothesis Hfix_boxed : fix_boxed fx = true.
Hypothesis Hfix_fields : fix_alias fx = true ∨ fix_deleted_field fx = true.
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
Variable loose : bool.
Variable dominated : list (node * node).
Variable to_unbox : list node.
Variable e : expr.

Local Abbreviation ctx := (plain_context is_local F T sets).
Local Abbreviation G := (graph_of_facts F).
Local Abbreviation sol := (solution_of_tables T).
Local Abbreviation cm := (U.bound_code_my_closures e).
Local Abbreviation unboxed :=
  (fst (U.unit_unboxed_fields F T dominated to_unbox e)).

Hypothesis Hleast : least_either is_local F T.
Hypothesis Hlocal : is_local_slots_only is_local.
Hypothesis Hsets : sets_of_graph G sets.
Hypothesis Hkinds : block_fields_of_one_kind G.
Hypothesis Henv : typing_env_satisfied_strictly statics valuation tenv.
Hypothesis Hfrag : env_in_fragment_with loose valuation tenv.
Hypothesis Halloc : statics_allocated sol statics.
Hypothesis Hst : statics_rebuilt G sol statics statics'.
Hypothesis Hcopies : copies_valued tenv canon valuation valuation'.
Hypothesis Hwhole : loose = true -> static_sets_whole sol statics statics'.
Hypothesis Huses : UCS.check_unboxed_uses F T unboxed = true.

Lemma unboxed_closed : closed is_local G sol.
Proof. exact (least_closed_tables is_local F T Hleast). Qed.

Local Abbreviation Hcl := unboxed_closed.

Lemma rw_sound_arbitrary (fuel : nat) (m : rewrite_metadata)
    (t : flambda_type) (v v' : value) :
  type_in_fragment_with loose valuation t -> rebuilt_under G sol m v v' ->
  type_satisfied statics valuation v t ->
  type_satisfied statics' valuation' v'
    (rewrite_arbitrary fx ctx tenv canon fuel m t).
Proof.
  intros Ht Hv Hsat.
  exact (proj1 (rewrite_sound fx Hfix_boxed Hfix_fields Hfix_refine Hfix_null
                  is_local F T sets tenv canon statics statics' valuation
                  valuation' loose Hcl Hleast Hlocal Hsets Henv Hfrag Halloc
                  Hst Hcopies Hwhole fuel)
           m t v v' Ht (or_introl Hv) Hsat).
Qed.

Lemma rw_forgets_unknown (fuel : nat) (m : rewrite_metadata)
    (k : flambda_kind) (w' : value) :
  forgets m = true -> kind_of_value w' = k ->
  type_satisfied statics' valuation' w'
    (rewrite_arbitrary fx ctx tenv canon fuel m (unknown k)).
Proof.
  intros Hm Hk.
  apply (rewrite_forgets_not_alias fx is_local F T sets tenv canon statics
           statics' valuation valuation' fuel m (unknown k) w' w' Hm);
    [ destruct k as [ | [] | | ]; reflexivity
    | apply unknown_satisfied; exact Hk
    | reflexivity ].
Qed.

Lemma unboxed_not_local_block (i : nat) (k : flambda_kind) :
  is_local (Field_block i k) = false.
Proof.
  destruct (is_local (Field_block i k)) eqn:H; [ | reflexivity ].
  destruct (Hlocal _ H) as [(w & Hw) | (g & Hg)]; discriminate.
Qed.

(* The type of a leaf, rewritten. *)
Definition leaf_rewritten (ty : flambda_type) (l : unboxed_leaf)
    : flambda_type :=
  let '(path, k, m) := l in
  match m with
  | None =>
    rewrite_arbitrary fx ctx tenv canon (rewrite_fuel tenv (unknown k))
      At_least_one_source_no_usages (unknown k)
  | Some m =>
    rewrite_arbitrary fx ctx tenv canon
      (rewrite_fuel tenv (destructure_path tenv path ty)) m
      (destructure_path tenv path ty)
  end.

(* A type at a level of the unboxing: what it expands to is of the
   fragment, and it holds of the value of the first run there, or is
   unknown. *)
Definition ty_ok (w : value) (ty : flambda_type) : Prop :=
  (∀ e0 : Expanded_type.t,
     expand_head tenv ty = Some e0 ->
     type_in_fragment_with loose valuation (Expanded_type.to_type e0))
  ∧ (type_satisfied statics valuation w ty ∨ ∃ k, ty = unknown k).

(* A level of the unboxing: a value at [n] comes from the allocation [b]
   only, [n] has a use, and [uses] are uses of [n] with a use, which are
   uses of names unboxed. *)
Definition level_ok (n b : node) (uses : list node) : Prop :=
  (∀ a : node, may_come_from sol a n -> s_has_source sol a = true -> a = b)
  ∧ s_has_usage sol n = true
  ∧ (s_any_usage sol n = true ∨ ∀ u : node, u ∈ uses -> s_usages sol n u = true)
  ∧ (∀ u : node, u ∈ uses -> s_has_usage sol u = true)
  ∧ (∀ u : node,
       u ∈ uses ->
       ∃ (x : node) (fs : U.unboxed_fields),
         (x, fs) ∈ unboxed ∧ mem_pair x u (t_usages T) = true).

(* The fields at a level are those that the analysis makes of the fields
   read from the uses. *)
Definition mk_shape (uses : list node) (fields : U.unboxed_fields) : Prop :=
  ∃ (fuel : nat) (block : node) (next : positive),
    fields
    = fst (U.mk_unboxed_fields F T cm dominated to_unbox fuel block
             (U.get_fields F T uses) next).

Lemma level_reads (n b : node) (uses : list node) :
  level_ok n b uses ->
  U.get_fields (rc_facts ctx) (rc_tables ctx)
    (U.add_usages_through_function_slots (rc_facts ctx) (rc_tables ctx) cm
       true uses)
  = U.get_fields (rc_facts ctx) (rc_tables ctx) uses.
Proof.
  intros (_ & _ & _ & _ & Hu).
  change (rc_facts ctx) with F. change (rc_tables ctx) with T.
  rewrite (add_usages_of_unboxed F T cm unboxed true uses Huses Hu).
  reflexivity.
Qed.

(* The uses of the level are uses of its allocation, or it has any
   usage. *)
Lemma level_uses (n b : node) (uses : list node) (u : node) :
  level_ok n b uses -> may_come_from sol b n -> u ∈ uses ->
  s_usages sol b u = true ∨ s_any_usage sol b = true.
Proof.
  intros (_ & Hhu & Hus & _) (_ & _ & _ & Hau & Hmu) Hu.
  destruct Hus as [Han | Hus]; [ right; exact (Hau Han) | ].
  exact (Hmu Hhu u (Hus u Hu)).
Qed.

(* A field read from the uses of a level is used by its allocation. *)
Lemma level_field_used (n b : node) (uses : list node) (f : field)
    (use : U.field_use) (from : node) :
  level_ok n b uses -> may_come_from sol b n -> g_constructor G b f from ->
  is_local f = false -> (f, use) ∈ U.get_fields F T uses ->
  s_field_used sol b f = true.
Proof.
  intros Hl Hm Hcons Hloc Hin.
  destruct (URP.get_fields_in F T uses f use Hin)
    as (y & u & Hacc & Hu & Hy).
  apply mem_node_In in Hu.
  destruct (level_uses n b uses u Hl Hm Hu) as [Hbu | Hab].
  - apply (field_used_usage _ _ _ Hcl b f from u y Hcons Hbu Hacc).
    destruct Hy as [Hy | Hy]; [ | exact Hy ].
    exact (has_usage_any _ _ _ Hcl y Hy).
  - exact (field_used_any_usage _ _ _ Hcl b f from Hcons Hab Hloc).
Qed.

(* The metadata of a leaf holds where its field is held. *)
Lemma level_metadata_at (n b : node) (uses : list node) (f : field)
    (use : U.field_use) (from : node) :
  level_ok n b uses -> may_come_from sol b n -> g_constructor G b f from ->
  is_local f = false -> (f, use) ∈ U.get_fields F T uses ->
  metadata_at sol (pattern_metadata ctx (get_single_field_source ctx b f) use)
    from.
Proof.
  intros Hl Hm Hcons Hloc Hin.
  pose proof (get_single_field_source_spec ctx b f) as Hs.
  cbn [plain_context rc_tables rc_facts] in Hs.
  pose proof (get_fields_use F T uses f use Hin) as Hu.
  unfold pattern_metadata.
  destruct (get_single_field_source ctx b f) as [ | s | ].
  - intros a Ha Hhs. exfalso. destruct (Hs from Hcons) as [Hnas Hno].
    destruct Ha as (_ & [Has | Hsrc] & _); [ congruence | ].
    exact (Hno a Hsrc Hhs).
  - intros a Ha Hhs. cbn [describes]. destruct (Hs from Hcons) as [Hnas Hone].
    destruct Ha as (_ & [Has | Hsrc] & _); [ congruence | ].
    exact (Hone a Hsrc Hhs).
  - (* the uses of what the field is read into *)
    assert (Hfrom_any : ∀ y u : node,
              g_accessor G y f u -> mem_node u uses = true ->
              s_any_usage sol b = false ->
              s_alias sol y from = true).
    { intros y u Hacc Hu' Hnb. apply mem_node_In in Hu'.
      destruct (level_uses n b uses u Hl Hm Hu') as [Hbu | Hab];
        [ | congruence ].
      exact (alias_constructor_usage _ _ _ Hcl b u f y from Hcons Hbu Hnb
               Hacc). }
    destruct use as [ | flow_to].
    + destruct Hu as (y & u & Hacc & Hu' & Hay).
      assert (Hafrom : s_any_usage sol from = true).
      { destruct (s_any_usage sol b) eqn:Hab.
        - exact (any_usage_constructor _ _ _ Hcl b f from Hab Hcons Hloc).
        - exact (any_usage_alias _ _ _ Hcl y from Hay
                   (Hfrom_any y u Hacc Hu' eq_refl)). }
      intros a Ha _. destruct Ha as (_ & _ & _ & Hau & _).
      cbn [describes]. exact (Hau Hafrom).
    + destruct Hu as (_ & Hfs & Hnot_any & Hys).
      intros a Ha _. cbn [describes].
      destruct (s_any_usage sol a) eqn:Haa; [ left; reflexivity | right ].
      destruct Ha as (_ & _ & _ & Hau & Hmu).
      assert (Hnf : s_any_usage sol from = false).
      { apply not_true_iff_false. intros H. rewrite (Hau H) in Haa.
        discriminate Haa. }
      assert (Hnb : s_any_usage sol b = false).
      { apply not_true_iff_false. intros H.
        rewrite (any_usage_constructor _ _ _ Hcl b f from H Hcons Hloc)
          in Hnf.
        discriminate Hnf. }
      intros z Hz. apply get_direct_usages_iff in Hz as (y & Hy & Hyz & Hhz).
      cbn [plain_context rc_tables rc_facts] in Hyz, Hhz.
      destruct (Hys y Hy) as (Hhy & u & Hacc & Hu').
      pose proof (Hfrom_any y u Hacc Hu' Hnb) as Halias.
      pose proof (usages_alias _ _ _ Hcl y from z Hyz
                    (Hnot_any y u Hacc Hu' Hfs) Halias Hnf) as Hfz.
      pose proof (has_usage_alias _ _ _ Hcl y from Hhy Halias) as Hhf.
      destruct (Hmu Hhf z Hfz) as [Haz | Haz]; [ | congruence ].
      split; [ exact Haz | exact Hhz ].
Qed.

Lemma expand_unknown (k : flambda_kind) :
  expand_head tenv (unknown k) = Some (Expanded_type.create_unknown k).
Proof. destruct k as [ | [] | | ]; reflexivity. Qed.

(* The alias to a name expands to a type of the fragment. *)
Lemma expand_alias_fragment (t : flambda_type) (s : simple)
    (e0 : Expanded_type.t) :
  get_alias_opt t = Some s -> expand_head tenv t = Some e0 ->
  type_in_fragment_with loose valuation (Expanded_type.to_type e0).
Proof.
  intros Ha. unfold expand_head. rewrite Ha.
  destruct (Typing_env.get_canonical_simple_exn tenv
              (Some Name_mode.in_types) None s)
    as [[c | ] | ]; cbn; [ | | discriminate ].
  - unfold expand_head_of_alias_type. destruct c as [x | sym | c].
    + destruct (Typing_env.find tenv (Name_var x) (Some (kind t)))
        as [ty | ] eqn:Hf; cbn; [ | discriminate ].
      intros He. exact (of_non_alias_fragment valuation ty e0 He
                          (find_fragment valuation tenv Hfrag _ _ ty Hf)).
    + destruct (Typing_env.find tenv (Name_symbol sym) (Some (kind t)))
        as [ty | ] eqn:Hf; cbn; [ | discriminate ].
      intros He. exact (of_non_alias_fragment valuation ty e0 He
                          (find_fragment valuation tenv Hfrag _ _ ty Hf)).
    + intros [= <-]. apply create_const_fragment.
  - intros He. apply (of_non_alias_fragment valuation _ e0 He).
    apply unknown_fragment.
Qed.

Lemma ty_ok_of (w : value) (ty : flambda_type) :
  type_in_fragment_with loose valuation ty ->
  (type_satisfied statics valuation w ty ∨ ∃ k, ty = unknown k) ->
  ty_ok w ty.
Proof.
  intros Hfr Hsat. split; [ | exact Hsat ].
  intros e0 He. exact (expand_head_fragment valuation tenv Hfrag ty e0 He Hfr).
Qed.

(* One field down. *)
Lemma ty_step (f : field) (w w1 : value) (ty : flambda_type) :
  ty_ok w ty -> field_of_value f w = Some w1 ->
  (∀ bn : boxable_number,
     f = Field_boxed_number bn -> kind_of_value w1 = unboxed_kind bn) ->
  type_in_fragment_with loose valuation
    (destructure_field f (expand_head tenv ty))
  ∧ (type_satisfied statics valuation w1
       (destructure_field f (expand_head tenv ty))
     ∨ destructure_field f (expand_head tenv ty) = unknown (U.field_kind f)).
Proof.
  intros [Hexp Hsat] Hfv Hbk.
  assert (Hua : unknown_accessor f = unknown (U.field_kind f)).
  { clear -Hfv. destruct f; cbn [field_of_value] in Hfv; try discriminate Hfv;
      reflexivity. }
  assert (Hunk : type_in_fragment_with loose valuation (unknown_accessor f)
                 ∧ (type_satisfied statics valuation w1 (unknown_accessor f)
                    ∨ unknown_accessor f = unknown (U.field_kind f)))
    by (rewrite Hua; split; [ apply unknown_fragment | right; reflexivity ]).
  destruct Hsat as [Hsat | (k & ->)].
  - destruct (expand_head tenv ty) as [e0 | ] eqn:He; [ | exact Hunk ].
    exact (destructure_field_sound statics valuation f e0 w w1
             (expand_head_sound statics valuation tenv ty e0 w Henv Hsat He)
             (Hexp e0 eq_refl) Hfv Hbk).
  - rewrite expand_unknown. exact Hunk.
Qed.

(* The leaves with no variable. *)
Lemma forgotten_sound (ty : flambda_type) (P : list (list field * flambda_kind))
    (ws : list value) (w_mid : value) :
  Forall2
    (fun (pk : list field * flambda_kind) (w' : value) =>
       value_at_path (fst pk) w_mid = Some w' ∧ kind_of_value w' = snd pk)
    P ws ->
  Forall2
    (fun (l : unboxed_leaf) (w' : value) =>
       type_satisfied statics' valuation' w' (leaf_rewritten ty l))
    (map (fun pk => (fst pk, snd pk, None)) P) ws.
Proof.
  induction 1 as [ | [p k] w' P ws [_ Hk] _ IH]; constructor; [ | exact IH ].
  cbn [leaf_rewritten fst snd] in *.
  exact (rw_forgets_unknown _ At_least_one_source_no_usages k w' eq_refl Hk).
Qed.

Lemma forgotten_leaves_eq (f : field) (u : U.unboxed_field) :
  forgotten_leaves f u
  = map (fun pk => (fst pk, snd pk, None))
      (map (fun pk => (f :: fst pk, snd pk))
         (unboxed_field_paths (U.field_kind f) u)).
Proof. unfold forgotten_leaves. rewrite map_map. reflexivity. Qed.


(* The names that the uses read from are uses of are unboxed. *)
Lemma unboxed_has (y : node) :
  U.has_to_be_unboxed dominated to_unbox y = true ->
  ∃ fs : U.unboxed_fields, (y, fs) ∈ unboxed.
Proof.
  intros Hy. unfold U.unit_unboxed_fields.
  exact (unboxed_fields_of_names_has F T cm dominated to_unbox _ y Hy).
Qed.

Lemma paths_step (f : field) (w_mid field' : value)
    (P : list (list field * flambda_kind)) (ws : list value) :
  field_of_value f w_mid = Some field' ->
  Forall2
    (fun (pk : list field * flambda_kind) (w' : value) =>
       value_at_path (fst pk) w_mid = Some w' ∧ kind_of_value w' = snd pk)
    (map (fun pk => (f :: fst pk, snd pk)) P) ws ->
  Forall2
    (fun (pk : list field * flambda_kind) (w' : value) =>
       value_at_path (fst pk) field' = Some w' ∧ kind_of_value w' = snd pk)
    P ws.
Proof.
  intros Hfv. revert ws. induction P as [ | pk P IH]; intros ws H;
    inversion H; subst; constructor; [ | exact (IH _ H4) ].
  destruct H2 as [Hv Hk]. cbn [value_at_path fst] in Hv. rewrite Hfv in Hv.
  split; [ exact Hv | exact Hk ].
Qed.

Lemma Forall2_prefix (f : field) (ty : flambda_type) (L : list unboxed_leaf)
    (ws : list value) :
  Forall2
    (fun (l : unboxed_leaf) (w' : value) =>
       type_satisfied statics' valuation' w'
         (leaf_rewritten (destructure_field f (expand_head tenv ty)) l))
    L ws ->
  Forall2
    (fun (l : unboxed_leaf) (w' : value) =>
       type_satisfied statics' valuation' w' (leaf_rewritten ty l))
    (map (fun l => let '(p, k, m) := l in (f :: p, k, m)) L) ws.
Proof.
  induction 1 as [ | [[p k] m] w' L ws H _ IH]; constructor; [ | exact IH ].
  destruct m; exact H.
Qed.

(* The level of a field unboxed in turn. *)
Lemma nested_level (n b : node) (uses : list node) (i : nat)
    (flow_to : list node) (s from : node) :
  level_ok n b uses -> may_come_from sol b n ->
  g_constructor G b (Field_block i Value) from ->
  s_has_usage sol from = true ->
  get_single_field_source ctx b (Field_block i Value)
  = Field_one_source s ->
  (Field_block i Value, U.Use_known flow_to) ∈ U.get_fields F T uses ->
  forallb (U.has_to_be_unboxed dominated to_unbox) flow_to = true ->
  level_ok from s (get_direct_usages ctx flow_to).
Proof.
  intros Hl Hmb Hcons Hhf Hsrc Hin Hall.
  pose proof (get_single_field_source_spec ctx b (Field_block i Value)) as Hs.
  rewrite Hsrc in Hs. cbn [plain_context rc_tables rc_facts] in Hs.
  destruct (Hs from Hcons) as [Hnas Hone].
  pose proof (get_fields_use F T uses _ _ Hin) as (_ & Hfs & Hnot_any & Hys).
  split; [ | split; [ exact Hhf | split; [ | split ] ] ].
  - intros a Ha Hha. destruct Ha as (_ & [Has | Hsrc'] & _);
      [ congruence | exact (Hone a Hsrc' Hha) ].
  - destruct (s_any_usage sol from) eqn:Haf; [ left; reflexivity | right ].
    assert (Hnb : s_any_usage sol b = false).
    { apply not_true_iff_false. intros H.
      rewrite (any_usage_constructor _ _ _ Hcl b _ from H Hcons
                 (unboxed_not_local_block i Value)) in Haf.
      discriminate Haf. }
    intros z Hz. apply get_direct_usages_iff in Hz as (y & Hy & Hyz & _).
    cbn [plain_context rc_tables rc_facts] in Hyz.
    destruct (Hys y Hy) as (_ & u & Hacc & Hu).
    pose proof Hu as Hu'. apply mem_node_In in Hu'.
    destruct (level_uses n b uses u Hl Hmb Hu') as [Hbu | Hab];
      [ | congruence ].
    pose proof (alias_constructor_usage _ _ _ Hcl b u _ y from Hcons Hbu
                  Hnb
                  Hacc) as Halias.
    exact (usages_alias _ _ _ Hcl y from z Hyz
             (Hnot_any y u Hacc Hu Hfs)
             Halias Haf).
  - intros z Hz. apply get_direct_usages_iff in Hz as (_ & _ & _ & Hhz).
    exact Hhz.
  - intros z Hz. apply get_direct_usages_iff in Hz as (y & Hy & Hyz & _).
    rewrite forallb_forall in Hall.
    destruct (unboxed_has y (Hall y Hy)) as (fs & Hfs').
    exists y, fs. split; [ exact Hfs' | exact Hyz ].
Qed.

(* The fields read from the uses of a name, as the analysis reads them. *)
Lemma nested_shape (flow_to : list node) (fuel : nat) (s : node)
    (next : positive) :
  forallb (U.has_to_be_unboxed dominated to_unbox) flow_to = true ->
  mk_shape (get_direct_usages ctx flow_to)
    (fst (U.mk_unboxed_fields F T cm dominated to_unbox fuel s
            (U.get_fields F T
               (U.add_usages_through_function_slots F T cm true
                  (U.get_direct_usages T flow_to)))
            next)).
Proof.
  intros Hall. exists fuel, s, next.
  rewrite (add_usages_of_unboxed F T cm unboxed true _ Huses).
  - rewrite (get_fields_ext F T (U.get_direct_usages T flow_to)
               (get_direct_usages ctx flow_to)); [ reflexivity | ].
    apply mem_node_iff_eq. intros z.
    rewrite (direct_usages_agree ctx flow_to z). reflexivity.
  - intros z Hz. apply (direct_usages_agree ctx) in Hz.
    apply get_direct_usages_iff in Hz as (y & Hy & Hyz & _).
    rewrite forallb_forall in Hall.
    destruct (unboxed_has y (Hall y Hy)) as (fs & Hfs').
    exists y, fs. split; [ exact Hfs' | exact Hyz ].
Qed.

Definition leaf_value_rel (w_mid : value) (pk : list field * flambda_kind)
    (w' : value)
    : Prop :=
  value_at_path (fst pk) w_mid = Some w' ∧ kind_of_value w' = snd pk.

(* The leaves, for a rewriting [rw] of the types of the leaves with a
   variable, sound of a value held in a field other than a function slot,
   and [rwn] of those with none: the rewriting of the result types, and
   that of the closures whose representation changes. *)
Section Leaves_with.

Variable rw : rewrite_metadata -> flambda_type -> flambda_type.
Variable rwn : flambda_kind -> flambda_type.

Hypothesis Hrwn : ∀ (k : flambda_kind) (w' : value),
  kind_of_value w' = k -> type_satisfied statics' valuation' w' (rwn k).

Hypothesis Hrw : ∀ (m : rewrite_metadata) (t : flambda_type) (v v' : value),
  type_in_fragment_with loose valuation t ->
  (∃ c : const, v = Value_const c ∧ v' = v)
  ∨ (∃ (a from : node) (f : field),
       g_constructor G a f from
       ∧ (∀ g : function_slot, f <> Field_function_slot g)
       ∧ rebuilt_value G sol v from v' ∧ metadata_at sol m from
       ∧ (forgets m = false -> s_has_usage sol from = true)) ->
  type_satisfied statics valuation v t ->
  type_satisfied statics' valuation' v' (rw m t).

Definition leaf_rewritten_with (ty : flambda_type) (l : unboxed_leaf)
    : flambda_type :=
  let '(path, k, m) := l in
  match m with
  | None => rwn k
  | Some m => rw m (destructure_path tenv path ty)
  end.

Definition leaf_type_rel_with (ty : flambda_type) (l : unboxed_leaf)
    (w' : value) : Prop :=
  type_satisfied statics' valuation' w' (leaf_rewritten_with ty l).

Lemma forgotten_sound_with (ty : flambda_type)
    (P : list (list field * flambda_kind)) (ws : list value) (w_mid : value)
    :
  Forall2 (leaf_value_rel w_mid) P ws ->
  Forall2 (leaf_type_rel_with ty) (map (fun pk => (fst pk, snd pk, None)) P)
    ws.
Proof.
  induction 1 as [ | [p k] w' P ws [_ Hk] _ IH]; constructor; [ | exact IH ].
  exact (Hrwn k w' Hk).
Qed.

Lemma Forall2_prefix_with (f : field) (ty : flambda_type)
    (L : list unboxed_leaf) (ws : list value) :
  Forall2 (leaf_type_rel_with (destructure_field f (expand_head tenv ty))) L
    ws ->
  Forall2 (leaf_type_rel_with ty)
    (map (fun l => let '(p, k, m) := l in (f :: p, k, m)) L) ws.
Proof.
  induction 1 as [ | [[p k] m] w' L ws H _ IH]; constructor; [ | exact IH ].
  destruct m; exact H.
Qed.

(* A leaf that is a field of the allocation of a level. *)
Lemma leaf_sound_with (n b : node) (uses : list node) (f : field)
    (use : U.field_use) (w w_mid w' : value) (ty : flambda_type) :
  level_ok n b uses -> rebuilt_value G sol w n w_mid -> ty_ok w ty ->
  (f, use) ∈ U.get_fields F T uses ->
  value_at_path [f] w_mid = Some w' -> kind_of_value w' = U.field_kind f ->
  type_satisfied statics' valuation' w'
    (rw (pattern_metadata ctx (get_single_field_source ctx b f) use)
       (destructure_field f (expand_head tenv ty))).
Proof.
  intros Hl Hrv Hty Hin Hv Hk. cbn [value_at_path] in Hv.
  destruct (field_of_value f w_mid) as [w1 | ] eqn:Hfv; [ | discriminate Hv ].
  injection Hv as ->.
  destruct f as [i k | | | | | | bn | | ]; cbn [field_of_value] in Hfv;
    try discriminate Hfv.
  - destruct w_mid as [ | | | | tag fields' | | | | ]; try discriminate Hfv.
    destruct (rebuilt_block_inv G sol w n tag fields' Hrv)
      as (fields & a & -> & Hma & Hha & Hfields).
    pose proof (proj1 Hl a Hma Hha) as ->.
    destruct (fields_rebuilt_nth _ fields fields' 0 i w' Hfields Hfv)
      as (field & Hn & Hfr).
    cbn [Nat.add] in Hfr.
    pose proof (field_rebuilt_kind G sol b i field w' Hfr) as Hkf.
    destruct Hfr as (from & Hcons & Hused & _).
    cbn [U.field_kind] in Hk. rewrite Hkf in Hk. subst k.
    pose proof (level_field_used n b uses _ use from Hl Hma Hcons
                  (unboxed_not_local_block _ _) Hin) as Hfu.
    destruct (Hused Hfu) as [Hhu Hrv'].
    destruct (ty_step (Field_block i (kind_of_value field))
                (Value_block tag fields) field ty Hty Hn
                (fun bn H => ltac:(discriminate H))) as [Hfr1 Hsat1].
    assert (Hsat : type_satisfied statics valuation field
                     (destructure_field (Field_block i (kind_of_value field))
                        (expand_head tenv ty))).
    { destruct Hsat1 as [Hsat1 | ->]; [ exact Hsat1 | ].
      apply unknown_satisfied. reflexivity. }
    apply (Hrw _ _ field w' Hfr1); [ | exact Hsat ].
    right. exists b, from, (Field_block i (kind_of_value field)).
    split; [ exact Hcons | split; [ intros g Hg; discriminate Hg | ] ].
    split; [ exact Hrv' | split; [ | intros _; exact Hhu ] ].
    exact (level_metadata_at n b uses _ use from Hl Hma Hcons
             (unboxed_not_local_block _ _) Hin).
  - destruct w_mid as [ | | | | | | c | | ]; try discriminate Hfv.
    injection Hfv as <-.
    pose proof (rebuilt_boxed_inv G sol w n c Hrv) as ->.
    destruct (ty_step (Field_boxed_number bn) (Value_boxed_number c)
                (Value_const c) ty Hty eq_refl
                (fun bn' H => ltac:(injection H as <-; exact Hk)))
      as [Hfr1 Hsat1].
    assert (Hsat : type_satisfied statics valuation (Value_const c)
                     (destructure_field (Field_boxed_number bn)
                        (expand_head tenv ty))).
    { destruct Hsat1 as [Hsat1 | Heq]; [ exact Hsat1 | rewrite Heq ].
      apply unknown_satisfied. exact Hk. }
    apply (Hrw _ _ (Value_const c) (Value_const c) Hfr1); [ | exact Hsat ].
    left. exists c. split; reflexivity.
Qed.

(* The leaves of every level. *)
Lemma leaves_sound_with :
  ∀ u : U.unboxed_field,
    match u with
    | U.Not_unboxed _ => True
    | U.Unboxed fields =>
      ∀ (kind : flambda_kind) (uses : list node) (b n : node)
        (w w_mid : value) (ty : flambda_type) (ws : list value),
        level_ok n b uses -> mk_shape uses fields ->
        rebuilt_value G sol w n w_mid -> ty_ok w ty ->
        Forall2 (leaf_value_rel w_mid)
          (unboxed_field_paths kind (U.Unboxed fields)) ws ->
        Forall2 (leaf_type_rel_with ty)
          (unboxed_leaves ctx cm kind uses b (U.Unboxed fields))
          ws
    end.
Proof.
  refine (URP.unboxed_field_rec _ _ _); [ intros; exact I | ].
  intros fields IH kind uses b n w w_mid ty ws Hl Hmk Hrv Hty Hp.
  destruct Hmk as (fuel & blk & next & Hfields).
  assert (Hnested :
    ∀ (f : field) (nested : U.unboxed_fields),
      (f, U.Unboxed nested) ∈ fields ->
      ∃ (flow_to : list node) (fuel' : nat) (source : node)
        (next' : positive),
        (f, U.Use_known flow_to) ∈ U.get_fields F T uses
        ∧ forallb (U.has_to_be_unboxed dominated to_unbox) flow_to = true
        ∧ nested
          = fst (U.mk_unboxed_fields F T cm dominated to_unbox fuel' source
                   (U.get_fields F T
                      (U.add_usages_through_function_slots F T cm true
                         (U.get_direct_usages T flow_to)))
                   next')).
  { intros f nested Hin. rewrite Hfields in Hin.
    exact (mk_unboxed_fields_nested F T cm dominated to_unbox fuel blk _ next
             f nested Hin). }
  clear Hfields fuel blk next.
  cbn [unboxed_leaves unboxed_field_paths] in Hp |- *.
  rewrite (level_reads n b uses Hl).
  set (fg := forgets_class _). clearbody fg.
  revert IH Hnested ws Hp.
  induction fields as [ | [f u] fields IHl]; intros IH Hnested ws Hp.
  { inversion Hp. constructor. }
  cbn beta iota in Hp |- *.
  apply Forall2_app_inv_l in Hp as (ws1 & ws2 & Hp1 & Hp2 & ->).
  apply Forall2_app.
  2: { apply IHl; [ intros f' u' H; exact (IH f' u' (or_intror H))
                  | intros f' n' H; exact (Hnested f' n' (or_intror H))
                  | exact Hp2 ]. }
  clear IHl Hp2.
  assert (Hforgot : Forall2 (leaf_type_rel_with ty) (forgotten_leaves f u)
                      ws1).
  { rewrite forgotten_leaves_eq.
    exact (forgotten_sound_with ty _ ws1 w_mid Hp1). }
  destruct (Keyed.lookup field_eqb f
              (U.get_fields (rc_facts ctx) (rc_tables ctx) uses))
    as [use | ] eqn:Hlk; [ | exact Hforgot ].
  apply (keyed_lookup_in field_eqb URP.field_eqb_decides) in Hlk.
  change (U.get_fields (rc_facts ctx) (rc_tables ctx) uses)
    with (U.get_fields F T uses) in Hlk.
  destruct fg; [ exact Hforgot | ].
  destruct u as [v | nested].
  - (* a leaf *)
    cbn [unboxed_field_paths map fst snd] in Hp1.
    inversion Hp1 as [ | pk w' P ws' [Hv Hk] Hnil]; subst.
    inversion Hnil; subst.
    constructor; [ | constructor ].
    exact (leaf_sound_with n b uses f use w w_mid w' ty Hl Hrv Hty Hlk Hv Hk).
  - (* a field unboxed in turn *)
    destruct use as [ | flow_to]; [ exact Hforgot | ].
    destruct (get_single_field_source ctx b f) as [ | s | ] eqn:Hsrc;
      try exact Hforgot.
    destruct ws1 as [ | w0 ws1].
    { pose proof (unboxed_leaves_paths ctx cm (U.Unboxed nested)
                    (U.field_kind f) (get_direct_usages ctx flow_to) s)
        as Hstrip.
      inversion Hp1 as [Hnil | ]. symmetry in Hnil.
      apply map_eq_nil in Hnil.
      change (unboxed_field_paths (U.field_kind f) (U.Unboxed nested) = [])
        in Hnil.
      rewrite Hnil in Hstrip. apply map_eq_nil in Hstrip.
      rewrite Hstrip. constructor. }
    destruct (unboxed_field_paths (U.field_kind f) (U.Unboxed nested))
      as [ | [p0 k0] P] eqn:Hpaths; [ inversion Hp1 | ].
    pose proof (unboxed_field_paths_cons (U.field_kind f) nested p0 k0)
      as Hp0. rewrite Hpaths in Hp0. specialize (Hp0 (or_introl eq_refl)).
    inversion Hp1 as [ | pk w' P' ws' [Hv0 _] _]; subst.
    cbn [value_at_path fst] in Hv0.
    destruct (field_of_value f w_mid) as [field' | ] eqn:Hfv;
      [ | discriminate Hv0 ].
    destruct p0 as [ | f2 p0]; [ contradiction | ].
    cbn [value_at_path] in Hv0.
    destruct (field_of_value f2 field') as [x2 | ] eqn:Hf2;
      [ clear Hv0 | discriminate Hv0 ].
    destruct f as [i k | | | | | | bn | | ]; cbn [field_of_value] in Hfv;
      try discriminate Hfv.
    2: { destruct w_mid; try discriminate Hfv. injection Hfv as <-.
         destruct f2; discriminate Hf2. }
    destruct w_mid as [ | | | | tag fields' | | | | ]; try discriminate Hfv.
    destruct (rebuilt_block_inv G sol w n tag fields' Hrv)
      as (fields0 & a & -> & Hma & Hha & Hfields0).
    pose proof (proj1 Hl a Hma Hha) as ->.
    destruct (fields_rebuilt_nth _ fields0 fields' 0 i field' Hfields0 Hfv)
      as (field & Hn & Hfr).
    cbn [Nat.add] in Hfr.
    destruct Hfr as (from & Hcons & Hused & Hunused).
    assert (Hfu : s_field_used sol b (Field_block i (kind_of_value field))
                  = true).
    { destruct (s_field_used sol b (Field_block i (kind_of_value field)))
        eqn:Hfu; [ reflexivity | ].
      rewrite (Hunused eq_refl) in Hf2. destruct f2; discriminate Hf2. }
    destruct (Hused Hfu) as [Hhf Hrv'].
    assert (Hkv : kind_of_value field = Value).
    { rewrite <- (rebuilt_value_kind G sol field from field' Hrv').
      destruct field'; try (destruct f2; discriminate Hf2); reflexivity. }
    rewrite Hkv in Hcons.
    destruct (get_single_field_source_one_constructor ctx b _ s Hsrc)
      as (from_k & Hcons_k).
    pose proof (Hkinds b from_k from i k Value Hcons_k Hcons) as ->.
    destruct (Hnested _ nested (or_introl eq_refl))
      as (flow_to' & fuel' & s' & next' & Hin' & Hall & Hnested_eq).
    pose proof (get_fields_functional F T uses _ _ _ Hlk Hin') as Heq.
    injection Heq as <-.
    pose proof (IH _ (U.Unboxed nested) (or_introl eq_refl)) as IHn.
    cbn beta iota in IHn.
    rewrite <- Hpaths in Hp1.
    apply Forall2_prefix_with.
    apply (IHn (U.field_kind (Field_block i Value))
             (get_direct_usages ctx flow_to) s from field field').
    + exact (nested_level n b uses i flow_to s from Hl Hma Hcons Hhf Hsrc
               Hlk Hall).
    + rewrite Hnested_eq. exact (nested_shape flow_to fuel' s' next' Hall).
    + exact Hrv'.
    + destruct (ty_step (Field_block i Value) (Value_block tag fields0) field
                  ty Hty Hn (fun bn H => ltac:(discriminate H)))
        as [Hfr1 Hsat1].
      apply (ty_ok_of field _ Hfr1).
      destruct Hsat1 as [Hsat1 | Heq]; [ left; exact Hsat1 | right ].
      exists Value. exact Heq.
    + exact (paths_step (Field_block i Value) (Value_block tag fields')
               field' _ _ Hfv Hp1).
Qed.

End Leaves_with.

(* The rewriting of the result types, as [Leaves_with] asks. *)
Definition rw_result (m : rewrite_metadata) (t : flambda_type) : flambda_type :=
  rewrite_arbitrary fx ctx tenv canon (rewrite_fuel tenv t) m t.

Definition rwn_result (k : flambda_kind) : flambda_type :=
  rewrite_arbitrary fx ctx tenv canon (rewrite_fuel tenv (unknown k))
    At_least_one_source_no_usages (unknown k).

Lemma rwn_result_sound (k : flambda_kind) (w' : value) :
  kind_of_value w' = k -> type_satisfied statics' valuation' w' (rwn_result k).
Proof.
  intros Hk.
  exact (rw_forgets_unknown _ At_least_one_source_no_usages k w' eq_refl Hk).
Qed.

Lemma rw_result_sound (m : rewrite_metadata) (t : flambda_type)
    (v v' : value) :
  type_in_fragment_with loose valuation t ->
  (∃ c : const, v = Value_const c ∧ v' = v)
  ∨ (∃ (a from : node) (f : field),
       g_constructor G a f from
       ∧ (∀ g : function_slot, f <> Field_function_slot g)
       ∧ rebuilt_value G sol v from v' ∧ metadata_at sol m from
       ∧ (forgets m = false -> s_has_usage sol from = true)) ->
  type_satisfied statics valuation v t ->
  type_satisfied statics' valuation' v' (rw_result m t).
Proof.
  intros Ht Hv Hsat. apply (rw_sound_arbitrary _ m t v v' Ht); [ | exact Hsat ].
  destruct Hv as [Hc | (a & from & f & _ & _ & H)]; [ left; exact Hc | ].
  right. exists from. exact H.
Qed.

Lemma leaf_rewritten_result (ty : flambda_type) (l : unboxed_leaf) :
  leaf_rewritten ty l = leaf_rewritten_with rw_result rwn_result ty l.
Proof. destruct l as [[p k] [m | ]]; reflexivity. Qed.

(* A leaf that is a field of the allocation of a level. *)
Lemma leaf_sound (n b : node) (uses : list node) (f : field)
    (use : U.field_use) (w w_mid w' : value) (ty : flambda_type) :
  level_ok n b uses -> rebuilt_value G sol w n w_mid -> ty_ok w ty ->
  (f, use) ∈ U.get_fields F T uses ->
  value_at_path [f] w_mid = Some w' -> kind_of_value w' = U.field_kind f ->
  type_satisfied statics' valuation' w'
    (rewrite_arbitrary fx ctx tenv canon
       (rewrite_fuel tenv (destructure_field f (expand_head tenv ty)))
       (pattern_metadata ctx (get_single_field_source ctx b f) use)
       (destructure_field f (expand_head tenv ty))).
Proof.
  exact (leaf_sound_with rw_result rw_result_sound n b uses f use w w_mid w'
           ty).
Qed.

Definition leaf_type_rel (ty : flambda_type) (l : unboxed_leaf) (w' : value)
    : Prop :=
  type_satisfied statics' valuation' w' (leaf_rewritten ty l).

(* The leaves of every level. *)
Lemma leaves_sound :
  ∀ u : U.unboxed_field,
    match u with
    | U.Not_unboxed _ => True
    | U.Unboxed fields =>
      ∀ (kind : flambda_kind) (uses : list node) (b n : node)
        (w w_mid : value) (ty : flambda_type) (ws : list value),
        level_ok n b uses -> mk_shape uses fields ->
        rebuilt_value G sol w n w_mid -> ty_ok w ty ->
        Forall2 (leaf_value_rel w_mid)
          (unboxed_field_paths kind (U.Unboxed fields)) ws ->
        Forall2 (leaf_type_rel ty)
          (unboxed_leaves ctx cm kind uses b (U.Unboxed fields))
          ws
    end.
Proof.
  intros u.
  pose proof (leaves_sound_with rw_result rwn_result rwn_result_sound
                rw_result_sound u) as H.
  destruct u as [v | fields]; [ exact I | ].
  intros kind uses b n w w_mid ty ws Hl Hmk Hrv Hty Hp.
  eapply Forall2_impl;
    [ | exact (H kind uses b n w w_mid ty ws Hl Hmk Hrv Hty Hp) ].
  intros l w' Hw. unfold leaf_type_rel. rewrite leaf_rewritten_result.
  exact Hw.
Qed.

(* A value with a field below it is of kind value. *)
Lemma path_kind_value (v v_mid w0 : value) (n : node) (f : field)
    (p : list field) :
  rebuilt_value G sol v n v_mid -> value_at_path (f :: p) v_mid = Some w0 ->
  kind_of_value v = Value.
Proof.
  intros Hrv Hp. cbn [value_at_path] in Hp.
  destruct (field_of_value f v_mid) as [x | ] eqn:Hfv; [ | discriminate Hp ].
  destruct f; cbn [field_of_value] in Hfv; try discriminate Hfv;
    destruct v_mid; try discriminate Hfv.
  - destruct (rebuilt_block_inv G sol v n _ _ Hrv) as (fs & a & -> & _).
    reflexivity.
  - rewrite (rebuilt_boxed_inv G sol v n _ Hrv). reflexivity.
Qed.

Lemma Forall2_sat_map (g : unboxed_leaf -> flambda_type)
    (leaves : list unboxed_leaf) (ws : list value) :
  Forall2 (fun l w' => type_satisfied statics' valuation' w' (g l)) leaves ws ->
  Forall2 (type_satisfied statics' valuation') ws (map g leaves).
Proof. induction 1; constructor; assumption. Qed.

Lemma rewrite_unboxed_result_type_sound_in (code_var type_var : variable)
    (fields : U.unboxed_fields) (v v_mid : value) (ws : list value) :
  Keyed.lookup node_eqb (Node_variable code_var) unboxed = Some fields ->
  lookup type_var valuation = Some v ->
  rebuilt_value G sol v (Node_variable code_var) v_mid ->
  unboxed_values fields v_mid ws ->
  Forall2 (type_satisfied statics' valuation') ws
    (rewrite_unboxed_result_type fx ctx tenv cm canon code_var type_var
       fields).
Proof.
  intros Hlk Hv Hrv Hws.
  assert (Heq : rewrite_unboxed_result_type fx ctx tenv cm canon code_var
                  type_var fields
                = map (leaf_rewritten (result_alias tenv type_var))
                    (unboxed_var_leaves ctx cm code_var fields)).
  { unfold rewrite_unboxed_result_type, unboxed_leaf_types. rewrite map_map.
    apply map_ext. intros [[p k] [m | ]]; reflexivity. }
  rewrite Heq. apply Forall2_sat_map.
  unfold unboxed_values in Hws.
  change (UCS.fields_paths fields)
    with (unboxed_field_paths Value (U.Unboxed fields)) in Hws.
  apply (keyed_lookup_in node_eqb node_eqb_decides) in Hlk.
  unfold unboxed_var_leaves. cbv zeta.
  pose proof (get_usages_spec ctx (Node_variable code_var)) as Hgus.
  pose proof (get_single_source_spec ctx (Node_variable code_var)) as Hgss.
  cbn [plain_context rc_tables rc_facts] in Hgus, Hgss.
  destruct (get_usages ctx (Node_variable code_var)) as [ | usages | ];
    [ exact (forgotten_sound _ _ ws v_mid Hws) | |
      exact (forgotten_sound _ _ ws v_mid Hws) ].
  destruct (get_single_source ctx (Node_variable code_var)) as [ | ap | ];
    [ exact (forgotten_sound _ _ ws v_mid Hws) | |
      exact (forgotten_sound _ _ ws v_mid Hws) ].
  destruct Hgus as (Hhu & Hnau & Hus). destruct Hgss as (Hhs & Hnas & Hone).
  destruct ws as [ | w0 ws].
  { inversion Hws as [Hnil | ]. symmetry in Hnil.
    change (unboxed_field_paths Value (U.Unboxed fields) = []) in Hnil.
    pose proof (unboxed_leaves_paths ctx cm (U.Unboxed fields) Value usages ap)
      as Hstrip.
    rewrite Hnil in Hstrip. apply map_eq_nil in Hstrip.
    rewrite Hstrip. constructor. }
  assert (Hkv : kind_of_value v = Value).
  { destruct (unboxed_field_paths Value (U.Unboxed fields))
      as [ | [p0 k0] P] eqn:Hpaths; [ inversion Hws | ].
    pose proof (unboxed_field_paths_cons Value fields p0 k0) as Hp0.
    rewrite Hpaths in Hp0. specialize (Hp0 (or_introl eq_refl)).
    inversion Hws as [ | pk w' P' ws' [Hv0 _] _]; subst.
    destruct p0 as [ | f0 p0]; [ contradiction | ].
    exact (path_kind_value v v_mid w0 _ f0 p0 Hrv Hv0). }
  apply (leaves_sound (U.Unboxed fields) Value usages ap
           (Node_variable code_var) v v_mid).
  - split; [ | split; [ exact Hhu | split; [ | split ] ] ].
    + intros a (_ & [Has | Hsrc] & _) Hha; [ congruence | ].
      exact (Hone a Hsrc Hha).
    + right. intros u Hu. exact (proj1 (proj1 (Hus u) Hu)).
    + intros u Hu. exact (proj2 (proj1 (Hus u) Hu)).
    + intros u Hu. exists (Node_variable code_var), fields.
      split; [ exact Hlk | exact (proj1 (proj1 (Hus u) Hu)) ].
  - destruct (URP.unboxed_fields_of_names_in F T cm dominated to_unbox _
                (Node_variable code_var) fields Hlk)
      as (a & next' & Hf).
    exists (UF.unboxing_depth T), a, next'. rewrite Hf.
    rewrite (add_usages_of_unboxed F T cm unboxed true _ Huses).
    + rewrite (get_fields_ext F T
                 (U.get_direct_usages T [Node_variable code_var]) usages);
        [ reflexivity | ].
      apply mem_node_iff_eq. intros z.
      rewrite <- (direct_usages_agree ctx [Node_variable code_var] z).
      rewrite get_direct_usages_iff, (Hus z). split.
      * intros (x & [<- | []] & Hxz & Hhz). split; assumption.
      * intros [Hxz Hhz]. exists (Node_variable code_var).
        split; [ left; reflexivity | split; assumption ].
    + intros z Hz. exists (Node_variable code_var), fields.
      split; [ exact Hlk | ].
      exact (proj1 (URP.get_direct_usages_in T _ z Hz)).
  - exact Hrv.
  - unfold result_alias. split.
    + intros e0 He.
      apply (expand_alias_fragment
               (alias_type_of (kind_of_name tenv (Name_var type_var))
                  (Var type_var))
               (Var type_var) e0); [ | exact He ].
      destruct (kind_of_name tenv (Name_var type_var)) as [ | [] | | ];
        reflexivity.
    + left. apply alias_satisfied. split; [ exact Hv | ].
      unfold kind_of_name.
      destruct (Typing_env.find tenv (Name_var type_var) None)
        as [ty0 | ] eqn:Hf; [ | exact Hkv ].
      exact (type_satisfied_kind _ _ _ _
               (find_satisfied_strictly statics valuation tenv
                  (Name_var type_var) None ty0 v Henv Hf Hv)).
  - exact Hws.
Qed.

End Unboxed_sound.

(* [unboxed_values] is the relation of the simulation of the unboxing. *)
Lemma unboxed_values_fields_vals (fields : U.unboxed_fields) (w : value)
    (ws : list value) :
  unboxed_values fields w ws
  <-> FlambdaReaperUnboxingRefinesCallsSimProofs.fields_vals fields w ws.
Proof. reflexivity. Qed.

Theorem rewrite_unboxed_result_type_sound :
  rewrite_unboxed_result_type_sound_statement.
Proof.
  intros is_local F T sets dominated to_unbox e tenv canon statics statics'
    valuation valuation' code_var type_var fields v v_mid ws Hleast Hlocal
    Hsets Hkinds Henv Hfrag Halloc Hst Hcopies Huses Hlk Hv Hrv Hws.
  assert (Hle : least_either is_local F T) by (left; exact Hleast).
  eapply (rewrite_unboxed_result_type_sound_in all_fixes eq_refl
            (or_introl eq_refl) eq_refl eq_refl is_local F T sets tenv canon
            statics statics' valuation valuation' false);
    try eassumption.
  intros H. discriminate H.
Qed.

Theorem rewrite_unboxed_result_type_sound_local :
  rewrite_unboxed_result_type_sound_local_statement.
Proof.
  intros is_local F T sets dominated to_unbox e tenv canon statics statics'
    valuation valuation' code_var type_var fields v v_mid ws Hleast Hlocal
    Hsets Hkinds Henv Hfrag Halloc Hst Hcopies Huses Hlk Hv Hrv Hws.
  assert (Hle : least_either is_local F T) by (right; exact Hleast).
  eapply (rewrite_unboxed_result_type_sound_in all_fixes eq_refl
            (or_introl eq_refl) eq_refl eq_refl is_local F T sets tenv canon
            statics statics' valuation valuation' false);
    try eassumption.
  intros H. discriminate H.
Qed.

Theorem rewrite_unboxed_result_type_sound_local_landed :
  rewrite_unboxed_result_type_sound_local_landed_statement.
Proof.
  intros is_local F T sets dominated to_unbox e tenv canon statics statics'
    valuation valuation' code_var type_var fields v v_mid ws Hleast Hlocal
    Hsets Hkinds Henv Hfrag Halloc Hst Hcopies Huses Hlk Hv Hrv Hws.
  assert (Hle : least_either is_local F T) by (right; exact Hleast).
  eapply (rewrite_unboxed_result_type_sound_in landed_fixes eq_refl
            (or_intror eq_refl) eq_refl eq_refl is_local F T sets tenv canon
            statics statics' valuation valuation' false);
    try eassumption.
  intros H. discriminate H.
Qed.

Theorem rewrite_unaliased_unboxed_result_type_sound :
  rewrite_unaliased_unboxed_result_type_sound_statement.
Proof.
  intros is_local F T sets dominated to_unbox e tenv canon statics statics'
    valuation valuation' code_var type_var fields v v_mid ws Hleast Hlocal
    Hsets Hkinds Henv Hfrag Halloc Hst Hwhole Hcopies Huses Hlk Hv Hrv Hws.
  assert (Hle : least_either is_local F T) by (left; exact Hleast).
  eapply (rewrite_unboxed_result_type_sound_in all_fixes eq_refl
            (or_introl eq_refl) eq_refl eq_refl is_local F T sets tenv canon
            statics statics' valuation valuation' true);
    try eassumption.
  intros _. exact Hwhole.
Qed.

Theorem unaliased_unboxed_result_type_sound_landed :
  unaliased_unboxed_result_type_sound_landed_statement.
Proof.
  intros is_local F T sets dominated to_unbox e tenv canon statics statics'
    valuation valuation' code_var type_var fields v v_mid ws Hleast Hlocal
    Hsets Hkinds Henv Hfrag Halloc Hst Hwhole Hcopies Huses Hlk Hv Hrv Hws.
  assert (Hle : least_either is_local F T) by (left; exact Hleast).
  eapply (rewrite_unboxed_result_type_sound_in landed_fixes eq_refl
            (or_intror eq_refl) eq_refl eq_refl is_local F T sets tenv canon
            statics statics' valuation valuation' true);
    try eassumption.
  intros _. exact Hwhole.
Qed.

Theorem rewrite_unaliased_unboxed_result_type_sound_local :
  rewrite_unaliased_unboxed_result_type_sound_local_statement.
Proof.
  intros is_local F T sets dominated to_unbox e tenv canon statics statics'
    valuation valuation' code_var type_var fields v v_mid ws Hleast Hlocal
    Hsets Hkinds Henv Hfrag Halloc Hst Hwhole Hcopies Huses Hlk Hv Hrv Hws.
  assert (Hle : least_either is_local F T) by (right; exact Hleast).
  eapply (rewrite_unboxed_result_type_sound_in all_fixes eq_refl
            (or_introl eq_refl) eq_refl eq_refl is_local F T sets tenv canon
            statics statics' valuation valuation' true);
    try eassumption.
  intros _. exact Hwhole.
Qed.

Theorem unaliased_unboxed_result_type_sound_local_landed :
  unaliased_unboxed_result_type_sound_local_landed_statement.
Proof.
  intros is_local F T sets dominated to_unbox e tenv canon statics statics'
    valuation valuation' code_var type_var fields v v_mid ws Hleast Hlocal
    Hsets Hkinds Henv Hfrag Halloc Hst Hwhole Hcopies Huses Hlk Hv Hrv Hws.
  assert (Hle : least_either is_local F T) by (right; exact Hleast).
  eapply (rewrite_unboxed_result_type_sound_in landed_fixes eq_refl
            (or_intror eq_refl) eq_refl eq_refl is_local F T sets tenv canon
            statics statics' valuation valuation' true);
    try eassumption.
  intros _. exact Hwhole.
Qed.

(* ======================================================================== *)
(* The exit relation and the rebuild                                        *)
(* ======================================================================== *)

From OxCamlFlambda2 Require FlambdaSemanticsProofs FlambdaReaperFlowProofs
  FlambdaReaperRebuildProofs.

Theorem naked_field_exit_relation : naked_field_exit_relation_statement.
Proof.
  split; [ apply (compute_least_sound _ _ 20); vm_compute; reflexivity | ].
  split; [ intros f Hf; discriminate Hf | ].
  split; [ intros f; reflexivity | ].
  do 8 (split; [ vm_compute; reflexivity | ]).
  split.
  - (* The graph has no constructor at a function slot. *)
    intros decls sites slot id Hl Hsites. exfalso.
    assert (Hb : is_bound slot decls = true)
      by (unfold is_bound; rewrite Hl; reflexivity).
    pose proof (proj2 (Hsites slot id Hl slot (sites slot))
                  (conj Hb eq_refl)) as Hcons.
    vm_compute in Hcons.
    destruct Hcons as [H | [H | [H | []]]]; discriminate H.
  - split; [ vm_compute; reflexivity | ].
    split; [ vm_compute; reflexivity | ].
    intros b Hrel.
    destruct (Hrel 1 NakedFieldExample.exit_state 1%positive
                [Value_symbol 50%positive] empty_heap no_reads
                ltac:(vm_compute; reflexivity) ltac:(vm_compute; reflexivity))
      as [_ (m & t & vs' & rd' & _ & _ & Hst)].
    destruct (Hst 50%positive
                (Static_block 0 [Value_const (Naked_immediate 3)])
                eq_refl ltac:(vm_compute; reflexivity))
      as (fields' & _ & Hfields).
    destruct fields' as [ | field' fields'];
      cbn [fields_rebuilt] in Hfields; [ exact Hfields | ].
    destruct Hfields as [(from & Hcons & _) _].
    vm_compute in Hcons.
    destruct Hcons as [H | [H | [H | []]]]; discriminate H.
Qed.

Theorem rebuild_exits_local : rebuild_exits_local_statement.
Proof.
  intros flag is_local u e T b n s k vs hp rd Hlocal Hfrag Hany Hrk Hwf
    Hexits Hcheck Hlrs Hexact Hrun Hstep.
  split.
  - destruct (FlambdaReaperFlowProofs.flow_sound_local flag is_local u e T n
                Hlocal
                (FlambdaReaperFlowProofs.in_flow_fragment_in_heap_fragment e
                   Hfrag)
                Hcheck Hlrs)
      as (located & Hflows).
    rewrite Hrun in Hflows. destruct Hflows as (_ & _ & Hst & _).
    intros x allocated Hl. exact (proj1 (Hst x allocated Hl)).
  - destruct (FlambdaReaperRebuildProofs.rebuild_refines_observably_local
                flag is_local u e T b Hlocal Hfrag Hany Hrk Hwf Hexits Hcheck
                Hlrs Hexact)
      as [Hfwd _].
    assert (Hb : behaves (State e [] [] empty_static_env empty_heap no_reads)
                   (Exits k vs hp)).
    { exists (S n), rd. replace (S n) with (n + 1) by lia.
      rewrite FlambdaSemanticsProofs.run_add, Hrun,
        FlambdaSemanticsProofs.run_succ, Hstep.
      reflexivity. }
    destruct (Hfwd k vs hp Hb) as (vs' & (m & rd' & Hm) & _).
    destruct (FlambdaReaperRebuildProofs.run_exit_state m _ k vs' hp rd' Hm)
      as (m' & t & Ht & Hexit).
    exists m', t, vs', rd'. split; assumption.
Qed.

Theorem rebuild_exits : rebuild_exits_statement.
Proof.
  intros flag is_local u e T b n s k vs hp rd Hlocal Hslots Hfrag Hany Hrk
    Hwf Hexits Hcheck Hexact.
  exact (rebuild_exits_local flag is_local u e T b n s k vs hp rd Hlocal Hfrag
           Hany Hrk Hwf Hexits Hcheck
           (FlambdaReaperFlowProofs.old_premise_local_reads_used flag
              is_local u e T Hlocal Hslots)
           Hexact).
Qed.

(* ------------------------------------------------------------------------ *)
(* From the relation of the simulation to [statics_rebuilt]                 *)
(* ------------------------------------------------------------------------ *)

Module RP := FlambdaReaperRebuildProofs.

Section Exit.

Variable is_local : field -> bool.
Variable G : graph.
Variable sol : solution.
Hypothesis Hc : closed is_local G sol.
Hypothesis Hexact : field_used_exact is_local G sol.

(* A field used holds what has a use, local or not. *)
Lemma used_field_hu (a from : node) (f : field) :
  s_field_used sol a f = true -> g_constructor G a f from ->
  s_has_usage sol from = true.
Proof.
  intros Hused Hcons.
  destruct (Hexact a f Hused)
    as (from0 & _ & [[Hau Hl] | [(usage & v & Hus & Hacc & Hhu)
                                | [(x & Hau & Hrd & Hhx)
                                  | (usage & v & Hsrc & Hl & Hau & Hacc
                                     & Hhu)]]]).
  - exact (has_usage_any _ _ _ Hc _
             (any_usage_constructor _ _ _ Hc _ _ _ Hau Hcons Hl)).
  - destruct (is_local f) eqn:Hl.
    + exact (has_usage_alias _ _ _ Hc _ _ Hhu
               (alias_local_constructor_usage _ _ _ Hc _ _ _ _ _ Hl Hcons
                  Hus Hacc)).
    + destruct (s_any_usage sol a) eqn:Ha.
      * exact (has_usage_any _ _ _ Hc _
                 (any_usage_constructor _ _ _ Hc _ _ _ Ha Hcons Hl)).
      * exact (has_usage_alias _ _ _ Hc _ _ Hhu
                 (alias_constructor_usage _ _ _ Hc _ _ _ _ _ Hcons Hus Ha
                    Hacc)).
  - destruct (is_local f) eqn:Hl.
    + exact (has_usage_alias _ _ _ Hc _ _ Hhx
               (alias_escaping_reading _ _ _ Hc _ _ _
                  (escaping_field_rule _ _ _ Hc _ _ _ Hau Hcons Hl) Hrd)).
    + exact (has_usage_any _ _ _ Hc _
               (any_usage_constructor _ _ _ Hc _ _ _ Hau Hcons Hl)).
  - exact (has_usage_alias _ _ _ Hc _ _ Hhu
             (alias_local_accessor_source _ _ _ Hc _ _ _ _ _ Hl Hacc Hsrc
                Hcons)).
Qed.

Variable kept : value_slot -> bool.
Variable called : node -> code_id -> Prop.
Variable exact partials : Prop.
Variable located : location -> option node.
Variable β : location -> location -> Prop.

Abbreviation rel := (RP.rel G sol kept called exact partials located β True).
Abbreviation slot_rel :=
  (RP.slot_rel G sol kept called exact partials located β True).

(* What the simulation has of the closures of a set, of which it relates
   a closure or a static closure, is what [rebuilt_value] and
   [static_rebuilt] ask of them, given what is asked of the value slots
   in turn ([P]). *)
Lemma closure_rebuilt (P : value -> node -> value -> Prop)
    (decls : list (function_slot * code_id)) (sites : function_slot -> node)
    (vs vs' : list (value_slot * value)) :
  RP.sites_exact G kept decls sites ->
  (∀ (f : function_slot) (id : code_id),
     lookup f decls = Some id ->
     (∀ (g : function_slot) (id' : code_id),
        lookup g decls = Some id' ->
        g_constructor G (sites f) (Field_function_slot g) (sites g))
     ∧ RP.slots_related (slot_rel (sites f) vs') vs []) ->
  (∀ (w : value_slot) (x : value) (from : node) (x' : value),
     lookup w vs = Some x -> rel x from x' ->
     (∀ c : const, x = Value_const c -> s_any_source sol from = true) ->
     P x from x') ->
  set_sites G decls sites
  ∧ for_all_bindings (value_slot_rebuilt G sol P decls sites vs') vs.
Proof.
  intros Hse Hall HP. split.
  - intros f id Hf g b. split.
    + intros Hb. exact (proj1 (Hse f id Hf) g b Hb).
    + intros [Hg ->]. destruct (proj1 (is_bound_lookup g decls) Hg)
        as [id' Hg'].
      exact (proj1 (Hall f id Hf) g id' Hg').
  - apply (proj2 (for_all_bindings_lookup _ _ vs)).
    intros w x Hl f id Hf Hused.
    destruct (Hexact _ _ Hused) as (from0 & Hcons0 & _).
    destruct (proj2 (Hse f id Hf) w from0 Hcons0) as [Hk Hheld].
    destruct (proj1 (RP.slots_related_iff _ _) (proj2 (Hall f id Hf)) w x Hl
                Hk)
      as (from & Hcons & Hhs & Has & Hfu).
    assert (Heq : from = from0) by exact (proj1 (Hheld f id from Hf) Hcons).
    subst from.
    pose proof (used_field_hu _ _ _ Hused Hcons) as Hhu.
    destruct (Hfu Hused) as (x' & Hx' & Hrel).
    exists from0, x'.
    split; [ intros g id' Hg b; exact (Hheld g id' b Hg) | ].
    split; [ exact Hhs | split; [ exact Hhu | split; [ exact Hx' | ] ] ].
    exact (HP w x from0 x' Hl (Hrel Hhu) Has).
Qed.

(* A value of the first run without a location, related by the simulation
   at [n] to one of the second, is rebuilt as [rebuilt_value] says, if
   [n] has any source where the value is a constant. *)
Lemma rel_rebuilt_value :
  ∀ (v : value) (n : node) (v' : value),
    rel v n v' -> locations_of_value v = [] ->
    (∀ c : const, v = Value_const c -> s_any_source sol n = true) ->
    rebuilt_value G sol v n v'.
Proof.
  apply (FlambdaHeapProofs.value_induction
           (fun v => ∀ (n : node) (v' : value),
              rel v n v' -> locations_of_value v = [] ->
              (∀ c : const, v = Value_const c ->
               s_any_source sol n = true) ->
              rebuilt_value G sol v n v'));
    cbn [RP.rel rebuilt_value locations_of_value].
  - intros c n v' [_ ->] _ Has. split; [ exact (Has c eq_refl) | reflexivity ].
  - intros code slots _ n v' [].
  - intros slot decls vs IH n v' (sites & vs' & Hmay & -> & _ & Hse & Hall)
      Hloc _.
    exists sites, vs'. split; [ reflexivity | split; [ exact Hmay | ] ].
    apply (closure_rebuilt (rebuilt_value G sol) decls sites vs vs'
             (Hse I)).
    + intros f id Hf. destruct (Hall f id Hf) as (_ & _ & Hfs & Hsr).
      exact (conj Hfs Hsr).
    + intros w x from x' Hl Hrel Has.
      rewrite Forall_forall in IH.
      apply (IH (w, x) (lookup_in _ _ _ Hl) from x' Hrel); [ | exact Has ].
      pose proof (proj1 (RP.flat_map_nil_iff _ _) Hloc) as Hl0.
      exact (Hl0 (w, x) (lookup_in _ _ _ Hl)).
  - intros s n v' (-> & Hmay & Hhu) _ _.
    split; [ reflexivity | split; [ exact Hmay | exact Hhu ] ].
  - intros tag fields IH n v' (a & fields' & Hmay & -> & Hf & Hhs & Hk)
      Hloc _.
    exists a, fields'.
    split; [ exact Hmay | split; [ exact Hhs | split; [ reflexivity | ] ] ].
    pose proof (proj1 (RP.flat_map_nil_iff _ _) Hloc) as Hl0.
    clear Hloc. rename Hl0 into Hloc.
    revert fields' Hf. generalize 0 as i.
    induction IH as [ | field fields Hfield _ IHf]; intros i [ | field' fields']
      Hf; cbn [RP.fields_related fields_rebuilt] in Hf |- *;
      try contradiction; [ exact I | ].
    apply Forall_cons_iff in Hk as [Hkf Hk].
    destruct Hf as [(from & Hcons & Hused & Hunused) Hrest].
    split.
    + rewrite Hkf. exists from.
      split; [ exact Hcons | split ].
      * intros Hu.
        split; [ exact (used_field_hu _ _ _ Hu Hcons) | ].
        apply (Hfield from field' (Hused Hu));
          [ exact (Hloc field (or_introl eq_refl)) | ].
        intros c ->. pose proof (Hused Hu) as Hr. cbn [RP.rel] in Hr.
        cbn [kind_of_value] in Hkf. rewrite Hkf in Hr.
        exact (proj1 Hr eq_refl).
      * exact Hunused.
    + exact (IHf Hk (fun v Hv => Hloc v (or_intror Hv)) (S i) fields'
               Hrest).
  - intros elements _ n v' (_ & elements' & -> & _) _ _.
    exists elements'. reflexivity.
  - intros c n v' (a & bn & from & Hmay & Hhs & Hk & Hcons & ->) _ _.
    exists a, bn, from. auto.
  - intros l n v' _ Hloc _. discriminate Hloc.
  - intros callee args arity _ _ n v' [_ (m & closure' & args' & -> & _)] _
      _.
    exists closure', args'. reflexivity.
Qed.

End Exit.

(* The static environments that the simulation relates where the unit
   exits ([RP.exit_statics_rel]) are related by [statics_rebuilt], if the
   first holds no location and its static values are of the fragment. *)
Lemma statics_rel_rebuilt (is_local : field -> bool) (G : graph)
    (sol : solution) (keep : value_slot -> bool)
    (statics statics' : static_env) :
  closed is_local G sol -> field_used_exact is_local G sol ->
  RP.exit_statics_rel G sol keep statics statics' ->
  RP.statics_no_location statics -> statics_of_values statics ->
  statics_rebuilt G sol statics statics'.
Proof.
  intros Hc Hexact (cds & F & b & ids & ff & cc & cdeps & cmds & ty & located
                    & Hnone & Hblocks & Honly & _)
    Hnl Hvals s allocated Hl Hhu.
  pose proof (Hvals s allocated Hl) as Hval.
  assert (Hloc : ∀ v : value, v ∈ FlambdaHeapSpec.values_of_static allocated ->
                 locations_of_value v = []).
  { intros v Hv. apply RP.no_location_iff. exact (Hnl s allocated v Hl Hv). }
  destruct (Honly s allocated Hl)
    as [(tag & fields & ->)
       | [(slot & syms & decls & vs & -> & Htarget & Hbound)
         | (Ho & Hbound)]].
  - destruct (Hblocks s tag fields Hl Hhu) as (fields' & Hl' & Hf).
    exists fields'. split; [ exact Hl' | ].
    cbn [FlambdaHeapSpec.values_of_static] in Hloc.
    clear Hl Hl' Hblocks Honly. revert fields' Hf. generalize 0 as i.
    induction fields as [ | field fields IH]; intros i [ | field' fields']
      Hf; cbn [RP.fields_related fields_rebuilt] in Hf |- *;
      try contradiction; [ exact I | ].
    destruct Hf as [(from & Hcons & Hused & Hunused) Hrest].
    assert (Hkf : kind_of_value field = Value)
      by exact (Hval field (or_introl eq_refl)).
    split.
    + unfold field_rebuilt. rewrite Hkf. exists from.
      split; [ exact Hcons | split; [ | exact Hunused ] ].
      intros Hu.
      split; [ exact (used_field_hu is_local G sol Hc Hexact _ _ _ Hu Hcons)
             | ].
      apply (rel_rebuilt_value is_local G sol Hc Hexact _ _ _ _ _ _ field
               from field' (Hused Hu));
        [ exact (Hloc field (or_introl eq_refl)) | ].
      intros c ->. pose proof (Hused Hu) as Hr. cbn [RP.rel] in Hr.
      cbn [kind_of_value] in Hkf. rewrite Hkf in Hr.
      exact (proj1 Hr eq_refl).
    + exact (IH (fun v Hv => Hloc v (or_intror Hv))
               (fun v Hv => Hval v (or_intror Hv)) (S i) fields' Hrest).
  - destruct Htarget as [Hn | (vs' & Hls' & _ & _ & Hsites)];
      [ destruct (Hbound Hhu Hn) | ].
    destruct (Hsites I) as (sites & Hsite & Hse & Hall).
    exists sites, vs'.
    split; [ exact Hls' | split; [ exact Hsite | ] ].
    apply (closure_rebuilt is_local G sol Hc Hexact _ _ _ _ _ _
             (rebuilt_value G sol) decls sites vs vs' Hse Hall).
    intros w x from x' Hlw Hrel Has.
    apply (rel_rebuilt_value is_local G sol Hc Hexact _ _ _ _ _ _ x from x'
             Hrel); [ | exact Has ].
    apply Hloc. cbn [FlambdaHeapSpec.values_of_static].
    apply in_map_iff. exists (w, x). split; [ reflexivity | ].
    exact (lookup_in _ _ _ Hlw).
  - destruct allocated as [ | | | contents | bytes | elements | ];
      cbn [static_rebuilt]; cbn [RP.static_other_rel] in Ho;
      try contradiction.
    + destruct Ho as [Hn | (v' & Hls' & Hv')];
        [ destruct (Hbound Hhu Hn) | ].
      rewrite Hls'. f_equal. f_equal.
      destruct Hv' as [Hv' | Hsim]; [ exact Hv' | ].
      destruct contents as [c | | | | | | | | ];
        cbn [kind_of_value boxable_number_of_kind] in Hval;
        try (exfalso; exact (Hval eq_refl)).
      exact Hsim.
    + destruct Ho as [Hn | Hls']; [ destruct (Hbound Hhu Hn) | exact Hls' ].
    + destruct Ho as [Hn | (elements' & Hls' & _)];
        [ destruct (Hbound Hhu Hn) | ].
      exists elements'. exact Hls'.
Qed.

(* [rebuild_exit_relation] of a unit whose closures are made where the
   graph has only the facts of their sets ([RP.closures_exact]) and whose
   static values are of the fragment where it exits. *)
Lemma exit_relation_of_exact_local (flag : reaper_preserve_direct_calls)
    (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool) :
  is_local_slots_only is_local ->
  in_flow_fragment e = true ->
  FlambdaReaperRebuildSpec.any_kind_arguments e = true ->
  FlambdaReaperRebuildSpec.results_kinded [] [] e = true ->
  well_formed_unit u e = true ->
  FlambdaReaperRebuildSpec.exits_one_value u e = true ->
  check_closed is_local (traverse_unit flag u e) T = true ->
  check_local_reads_used is_local (traverse_unit flag u e) T = true ->
  check_field_used_exact is_local (traverse_unit flag u e) T = true ->
  RP.closures_exact (graph_of_facts (traverse_unit flag u e)) (unit_kept e)
    e ->
  exits_of_values e ->
  rebuild_exit_relation (graph_of_facts (traverse_unit flag u e))
    (solution_of_tables T) e
    (rebuild_unit_from_tables (traverse_unit flag u e) T
       preserve_direct_calls e).
Proof.
  intros Hlocal Hfrag Hdc Hrk Hwf Hexits Hcheck Hlrs Hcheck_exact Hxe Hvals
    n s k vs hp rd Hrun Hstep.
  split.
  - exact (proj1 (rebuild_exits_local flag is_local u e T
                    preserve_direct_calls n s k vs hp rd Hlocal Hfrag Hdc Hrk
                    Hwf Hexits Hcheck Hlrs Hcheck_exact Hrun Hstep)).
  - destruct (RP.rebuild_exit_statics_local flag is_local u e T
                preserve_direct_calls Hlocal Hfrag Hdc Hrk Hwf Hexits Hcheck
                Hlrs Hcheck_exact Hxe n s k vs hp rd Hrun Hstep)
      as (m & t & vs' & rd' & Hm & Ht & Hst & Hnl).
    exists m, t, vs', rd'.
    split; [ exact Hm | split; [ exact Ht | ] ].
    exact (statics_rel_rebuilt is_local _ _ _ _ _
             (check_closed_sound is_local _ _ Hcheck)
             (check_field_used_exact_sound is_local _ _ Hcheck_exact) Hst Hnl
             (Hvals n s k vs hp rd Hrun Hstep)).
Qed.

Lemma exit_relation_of_exact (flag : reaper_preserve_direct_calls)
    (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool) :
  is_local_slots_only is_local ->
  (∀ f : field, is_local f = false) ∨ projection_free e = true ->
  in_flow_fragment e = true ->
  FlambdaReaperRebuildSpec.any_kind_arguments e = true ->
  FlambdaReaperRebuildSpec.results_kinded [] [] e = true ->
  well_formed_unit u e = true ->
  FlambdaReaperRebuildSpec.exits_one_value u e = true ->
  check_closed is_local (traverse_unit flag u e) T = true ->
  check_field_used_exact is_local (traverse_unit flag u e) T = true ->
  RP.closures_exact (graph_of_facts (traverse_unit flag u e)) (unit_kept e)
    e ->
  exits_of_values e ->
  rebuild_exit_relation (graph_of_facts (traverse_unit flag u e))
    (solution_of_tables T) e
    (rebuild_unit_from_tables (traverse_unit flag u e) T
       preserve_direct_calls e).
Proof.
  intros Hlocal Hslots Hfrag Hdc Hrk Hwf Hexits Hcheck Hcheck_exact.
  exact (exit_relation_of_exact_local flag is_local u e T preserve_direct_calls
           Hlocal Hfrag Hdc Hrk Hwf Hexits Hcheck
           (FlambdaReaperFlowProofs.old_premise_local_reads_used flag
              is_local u e T Hlocal Hslots)
           Hcheck_exact).
Qed.

From OxCamlFlambda2 Require FlambdaReaperClosuresExactProofs.

Theorem rebuild_exit_relation_holds_local :
  rebuild_exit_relation_holds_local_statement.
Proof.
  intros flag is_local u e T b Hlocal Hfrag Hdc Hrk Hwf Hexits Hcheck Hlrs
    Hcheck_exact Honce Hvals.
  apply (exit_relation_of_exact_local flag is_local u e T b Hlocal Hfrag Hdc
           Hrk Hwf Hexits Hcheck Hlrs Hcheck_exact); [ | exact Hvals ].
  exact (FlambdaReaperClosuresExactProofs.closures_exact_of_bound_once flag u
           e Hfrag Hwf Honce).
Qed.

Theorem rebuild_exit_relation_holds : rebuild_exit_relation_holds_statement.
Proof.
  intros flag is_local u e T b Hlocal Hslots Hfrag Hdc Hrk Hwf Hexits Hcheck
    Hcheck_exact.
  exact (rebuild_exit_relation_holds_local flag is_local u e T b Hlocal Hfrag
           Hdc Hrk Hwf Hexits Hcheck
           (FlambdaReaperFlowProofs.old_premise_local_reads_used flag
              is_local u e T Hlocal Hslots)
           Hcheck_exact).
Qed.

Theorem exported_types_sound_end_to_end_local :
  exported_types_sound_end_to_end_local_statement.
Proof.
  intros flag is_local u e T b sets tenv canon valuation valuation' n s k vs
    hp rd sym allocated Hlocal Hfrag Hdc Hrk Hwf Hexits Hleast Honce Hvals
    Hsets Hrun Hstep Henv Hfr Hcopies Hl.
  destruct (least_local_checks is_local _ T Hleast) as [Hcheck Hlrs].
  apply (exported_types_sound_local flag is_local u e T b sets tenv canon
           valuation valuation' n s k vs hp rd sym allocated Hlocal Hleast
           Hsets);
    try assumption.
  exact (rebuild_exit_relation_holds_local flag is_local u e T b Hlocal Hfrag
           Hdc Hrk Hwf Hexits Hcheck Hlrs
           (least_field_used_exact_local is_local _ T Hleast) Honce Hvals).
Qed.

Theorem exported_types_sound_end_to_end_local_landed :
  exported_types_sound_end_to_end_local_landed_statement.
Proof.
  intros flag is_local u e T b sets tenv canon valuation valuation' n s k vs
    hp rd sym allocated Hlocal Hfrag Hdc Hrk Hwf Hexits Hleast Honce Hvals
    Hsets Hrun Hstep Henv Hfr Hcopies Hl.
  destruct (least_local_checks is_local _ T Hleast) as [Hcheck Hlrs].
  apply (exported_types_sound_local_landed flag is_local u e T b sets tenv canon
           valuation valuation' n s k vs hp rd sym allocated Hlocal Hleast
           Hsets);
    try assumption.
  exact (rebuild_exit_relation_holds_local flag is_local u e T b Hlocal Hfrag
           Hdc Hrk Hwf Hexits Hcheck Hlrs
           (least_field_used_exact_local is_local _ T Hleast) Honce Hvals).
Qed.

Theorem exported_types_sound_end_to_end :
  exported_types_sound_end_to_end_statement.
Proof.
  intros flag is_local u e T b sets tenv canon valuation valuation' n s k vs
    hp rd sym allocated Hlocal Hslots Hfrag Hdc Hrk Hwf Hexits Hleast.
  exact (exported_types_sound_end_to_end_local flag is_local u e T b sets tenv
           canon valuation valuation' n s k vs hp rd sym allocated Hlocal Hfrag
           Hdc Hrk Hwf Hexits
           (least_local_of_least is_local _ T Hleast
              (FlambdaReaperFlowProofs.old_premise_local_reads_used flag
                 is_local u e T Hlocal Hslots))).
Qed.

Lemma exported_types_sound_end_to_end_gen (fx : fixes) :
  fixes_sound fx -> exported_types_sound_end_to_end_with fx.
Proof.
  intros Hfx flag is_local u e T b sets tenv canon valuation valuation' n s k
    vs hp rd sym allocated Hlocal Hslots Hfrag Hdc Hrk Hwf Hexits Hleast Honce
    Hvals Hsets Hrun Hstep Henv Hfr Hcopies Hl.
  pose proof (least_local_of_least is_local _ T Hleast
                (FlambdaReaperFlowProofs.old_premise_local_reads_used flag
                   is_local u e T Hlocal Hslots)) as Hll.
  destruct (least_local_checks is_local _ T Hll) as [Hcheck Hlrs].
  apply (exported_types_sound_gen fx Hfx flag is_local u e T b sets tenv canon
           valuation valuation' n s k vs hp rd sym allocated Hlocal Hleast
           Hsets);
    try assumption.
  exact (rebuild_exit_relation_holds_local flag is_local u e T b Hlocal Hfrag
           Hdc Hrk Hwf Hexits Hcheck Hlrs
           (least_field_used_exact_local is_local _ T Hll) Honce Hvals).
Qed.

(* ------------------------------------------------------------------------ *)
(* The static sets kept whole                                               *)
(* ------------------------------------------------------------------------ *)

From OxCamlFlambda2 Require FlambdaClosuresSpec FlambdaClosuresProofs
  FlambdaWellFormedProofs.

(* The rebuild keeps the sets of a term well-formed: it keeps the
   function slots of a set, and keeps or removes the bindings of a
   static set together, as their symbols ([static_const_is_used]). *)
Section Rebuilt_sets.

Variable F : facts.
Variable sol : solution.
Variable b : bool.
Variable ids : list code_id.
Variable ff : positive.

Local Abbreviation rebuild := (rebuild_expr F sol b ids ff None).
Local Abbreviation rebuild_h := (rebuild_handler F sol b ids ff None).
Local Abbreviation rebuild_b := (rebuild_binding F sol b ids ff None).

(* A binding rebuilt has the code ids and the symbols of the sets of the
   binding. *)
Lemma rebuild_binding_shape (en : rebuild_env) (bd bd' : static_binding) :
  bd' ∈ rebuild_b en bd ->
  static_set_code_ids bd' = static_set_code_ids bd
  ∧ (∀ c : static_binding,
       binds_symbol_of_set bd' c = binds_symbol_of_set bd c)
  ∧ ∀ c : static_binding,
      binds_symbol_of_set c bd' = binds_symbol_of_set c bd.
Proof.
  destruct bd as [id c | id | s const]; cbn [rebuild_binding].
  - destruct (is_code_id_used sol id); intros [<- | []];
      (split; [ reflexivity | split ]); intros c';
      destruct c' as [ | | s' [ ] ]; reflexivity.
  - intros [<- | []]. auto.
  - destruct (static_const_is_used sol s const); [ | intros [] ].
    intros [<- | []].
    split; [ destruct const; reflexivity | split ]; intros c';
      destruct const; destruct c' as [ | | s' [ ] ]; reflexivity.
Qed.

(* [static_sets_code_disjoint] of a group whose bindings are mapped to
   at most one each, of the same shape. *)
Lemma code_disjoint_flat_map
    (f : static_binding -> list static_binding) :
  (∀ bd : static_binding, f bd = [] ∨ ∃ bd', f bd = [bd']) ->
  (∀ bd bd' : static_binding,
     bd' ∈ f bd ->
     static_set_code_ids bd' = static_set_code_ids bd
     ∧ (∀ c : static_binding,
          binds_symbol_of_set bd' c = binds_symbol_of_set bd c)
     ∧ ∀ c : static_binding,
         binds_symbol_of_set c bd' = binds_symbol_of_set c bd) ->
  ∀ l : list static_binding,
    static_sets_code_disjoint l = true ->
    static_sets_code_disjoint (flat_map f l) = true.
Proof.
  intros Hone Hshape. induction l as [ | bd l IH]; [ reflexivity | ].
  cbn [static_sets_code_disjoint flat_map].
  intros H. apply andb_true_iff in H as [Hall Hl].
  destruct (Hone bd) as [-> | (bd' & Hbd)]; [ exact (IH Hl) | ].
  rewrite Hbd. cbn [app static_sets_code_disjoint].
  rewrite (IH Hl), andb_true_r.
  destruct (Hshape bd bd' (ltac:(rewrite Hbd; left; reflexivity)))
    as (Hcodes & Hsym & _).
  apply forallb_forall. intros b2' Hin.
  apply in_flat_map in Hin as (b2 & Hb2 & Hin).
  destruct (Hshape b2 b2' Hin) as (Hcodes2 & _ & Hsym2).
  rewrite Hsym, Hsym2, Hcodes, Hcodes2.
  exact (proj1 (forallb_forall _ _) Hall b2 Hb2).
Qed.

Lemma rebuild_bindings_whole (en : rebuild_env)
    (bindings : list static_binding) :
  forallb (static_set_whole bindings) bindings = true ->
  forallb (static_set_whole (flat_map (rebuild_b en) bindings))
    (flat_map (rebuild_b en) bindings)
  = true.
Proof.
  intros H. apply forallb_forall. intros bd' Hin'.
  apply in_flat_map in Hin' as (bd & Hin & Hbd').
  pose proof (proj1 (forallb_forall _ _) H bd Hin) as Hw.
  destruct bd as [id c | id | s const]; cbn [rebuild_binding] in Hbd'.
  - destruct (is_code_id_used sol id); destruct Hbd' as [<- | []];
      reflexivity.
  - destruct Hbd' as [<- | []]. reflexivity.
  - destruct (static_const_is_used sol s const) eqn:Hused;
      [ | destruct Hbd' ].
    destruct Hbd' as [<- | []].
    destruct const as [code vss | slot syms decls vss | tag fields | contents
                      | bytes | elements];
      try reflexivity.
    cbn [rebuild_static_const static_set_whole] in Hw |- *.
    apply andb_true_iff in Hw as [Hw Hall]. rewrite Hw. cbn [andb].
    apply forallb_forall. intros [slot' s'] Hin_s.
    pose proof (proj1 (forallb_forall _ _) Hall (slot', s') Hin_s) as Hex.
    cbn beta iota in Hex.
    apply existsb_exists in Hex as (bd2 & Hbd2 & Hbinds).
    apply FlambdaClosuresProofs.binds_symbol_to_sound in Hbinds.
    subst bd2.
    apply existsb_exists.
    eexists. split.
    + apply in_flat_map.
      exists (Bind_symbol s' (Static_const_closure_of_set slot' syms decls
                                vss)).
      split; [ exact Hbd2 | ].
      cbn [rebuild_binding]. cbn [static_const_is_used] in Hused |- *.
      rewrite Hused. left. reflexivity.
    + cbn [rebuild_static_const binds_symbol_to]. rewrite Pos.eqb_refl.
      destruct (static_const_eq_dec _ _) as [_ | Hne];
        [ reflexivity | exfalso; apply Hne; reflexivity ].
Qed.

Lemma rebuild_let_static_unfold (en : rebuild_env)
    (bindings : list static_binding) (body : expr) :
  rebuild en (Let_static bindings body)
  = Let_static (flat_map (rebuild_b en) bindings)
      (rebuild (bind_variables en []) body).
Proof. reflexivity. Qed.

Lemma sets_well_formed_let_static (bindings : list static_binding)
    (body : expr) :
  sets_well_formed (Let_static bindings body)
  = forallb (static_set_whole bindings) bindings
    && static_sets_code_disjoint bindings
    && forallb binding_sets_well_formed bindings
    && sets_well_formed body.
Proof. reflexivity. Qed.

Lemma rebuild_sets_well_formed :
  (∀ (e : expr) (en : rebuild_env),
     sets_well_formed e = true -> sets_well_formed (rebuild en e) = true)
  ∧ ∀ (h : continuation_handler) (en : rebuild_env) (dc : bool)
      (mask : list bool),
      handler_sets_well_formed h = true ->
      handler_sets_well_formed (rebuild_h en dc mask h) = true.
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - intros x kind d body IH en H. cbn [sets_well_formed] in H.
    apply andb_true_iff in H as [Hd Hb].
    cbn [rebuild_expr].
    destruct (is_pending_projection en x d).
    + cbn [sets_well_formed]. rewrite Hd. exact (IH _ Hb).
    + destruct d; cbn [named_sets_well_formed] in Hd;
        repeat match goal with
               | |- sets_well_formed (if ?c then _ else _) = true =>
                 destruct c
               end;
        try reflexivity; try exact (IH _ Hb);
        cbn [sets_well_formed]; rewrite (IH _ Hb), andb_true_r;
        [ .. | exact Hd ];
        unfold rebuild_named;
        repeat match goal with
               | |- named_sets_well_formed (match ?c with _ => _ end) = true
                 => destruct c
               | |- named_sets_well_formed (if ?c then _ else _) = true =>
                 destruct c
               end;
        reflexivity.
  - intros bindings body Hcodes IH en H.
    rewrite sets_well_formed_let_static in H.
    apply andb_true_iff in H as [H Hb].
    apply andb_true_iff in H as [H Hbs].
    apply andb_true_iff in H as [Hwhole Hdisj].
    rewrite rebuild_let_static_unfold, sets_well_formed_let_static.
    rewrite (rebuild_bindings_whole en bindings Hwhole).
    rewrite (code_disjoint_flat_map (rebuild_b en)); [ | | | exact Hdisj ].
    + rewrite (IH _ Hb), !andb_true_r. cbn [andb].
      apply forallb_forall. intros bd' Hin'.
      apply in_flat_map in Hin' as (bd & Hin & Hbd').
      pose proof (proj1 (forallb_forall _ _) Hbs bd Hin) as Hw.
      destruct bd as [id c | id | s const]; cbn [rebuild_binding] in Hbd'.
      * destruct (is_code_id_used sol id); destruct Hbd' as [<- | []];
          [ | reflexivity ].
        pose proof (Hcodes id c Hin) as Hc.
        destruct c as [params mc ret exn md cbody].
        cbn [binding_sets_well_formed code_sets_well_formed code_body
             rebuild_code] in Hw, Hc |- *.
        destruct (existsb _ _); [ reflexivity | exact (Hc _ Hw) ].
      * destruct Hbd' as [<- | []]. reflexivity.
      * destruct (static_const_is_used sol s const); [ | destruct Hbd' ].
        destruct Hbd' as [<- | []]. reflexivity.
    + intros bd. destruct bd as [id c | id | s const]; cbn [rebuild_binding].
      * destruct (is_code_id_used sol id); right; eexists; reflexivity.
      * right. eexists. reflexivity.
      * destruct (static_const_is_used sol s const);
          [ right; eexists; reflexivity | left; reflexivity ].
    + exact (rebuild_binding_shape en).
  - intros k h body IHh IH en H. cbn [sets_well_formed] in H.
    apply andb_true_iff in H as [Hh Hb].
    cbn [rebuild_expr].
    destruct (among k _); [ | exact (IH _ Hb) ].
    cbn [sets_well_formed]. rewrite (IHh _ _ _ Hh). exact (IH _ Hb).
  - intros handlers body IHh IH en H. cbn [sets_well_formed] in H.
    apply andb_true_iff in H as [Hh Hb].
    cbn [rebuild_expr sets_well_formed]. rewrite (IH _ Hb), andb_true_r.
    apply forallb_forall. intros [k h'] Hin.
    apply in_map_iff in Hin as ([k0 h0] & Heq & Hin). injection Heq as <- <-.
    apply IHh with (k := k0); [ exact Hin | ].
    exact (proj1 (forallb_forall _ _) Hh (k0, h0) Hin).
  - intros ap en _. cbn [rebuild_expr]. unfold rebuild_apply.
    destruct (_ || _); [ reflexivity | ]. unfold make_apply_wrapper.
    destruct (lookup _ _); [ destruct (forallb _ _) | ]; reflexivity.
  - intros ac en _. cbn [rebuild_expr]. unfold rebuild_apply_cont.
    destruct (rewrite_apply_cont _ _ _); reflexivity.
  - intros scrutinee arms en _. cbn [rebuild_expr]. unfold rebuild_switch.
    destruct (Keyed.is_empty _); reflexivity.
  - intros en _. reflexivity.
  - intros params metadata handler IH en dc mask H.
    cbn [rebuild_handler handler_sets_well_formed] in H |- *.
    destruct (dc && _); [ reflexivity | exact (IH _ H) ].
Qed.

End Rebuilt_sets.

(* The rebuilt unit of a well-formed unit binds its static sets whole
   wherever it runs to. *)
Lemma rebuilt_sets_bound (F : facts) (T : tables) (b : bool)
    (u : unit_interface) (e : expr) (m : nat) (t : state) :
  well_formed_unit u e = true ->
  run m
    (State (rebuild_unit_from_tables F T b e) [] [] empty_static_env
       empty_heap no_reads)
  = Step t ->
  FlambdaClosuresSpec.static_sets_bound (state_statics t).
Proof.
  intros Hwf Hm.
  assert (Hsets : sets_well_formed (rebuild_unit_from_tables F T b e) = true).
  { unfold rebuild_unit_from_tables, rebuild_unit.
    apply (proj1 (rebuild_sets_well_formed F (solution_of_tables T) b _ _)).
    exact (FlambdaWellFormedProofs.well_formed_unit_sets_well_formed u e
             Hwf). }
  exact (proj2 (proj2 (proj2
           (FlambdaClosuresProofs.sets_inv_run m _ t
              (FlambdaClosuresProofs.sets_inv_initial _ Hsets) Hm)))).
Qed.

(* The static environments that the simulation relates where the unit
   exits keep the static sets whole, if each run binds the symbols of
   its sets together: a symbol of a set that has a use is bound in the
   second run, to a closure of the same set. *)
Lemma statics_rel_whole (G : graph) (sol : solution)
    (keep : value_slot -> bool) (statics statics' : static_env) :
  RP.exit_statics_rel G sol keep statics statics' ->
  FlambdaClosuresSpec.static_sets_bound statics ->
  FlambdaClosuresSpec.static_sets_bound statics' ->
  static_sets_whole sol statics statics'.
Proof.
  intros (cds & F & b & ids & ff & cc & cdeps & cmds & ty & located & _ & _
          & Honly & _)
    Hbound Hbound' s slot syms decls vs Hl Hhu.
  destruct (Hbound s slot syms decls vs Hl) as [Hslot Hset].
  split; [ exact Hslot | ].
  destruct (Honly s _ Hl)
    as [(tag & fields & Heq)
       | [(slot0 & syms0 & decls0 & vs0 & Heq & Htarget & Hbinds)
         | (Ho & _)]];
    [ discriminate Heq | | destruct Ho ].
  injection Heq as <- <- <- <-.
  destruct Htarget as [Hn | (vs' & Hls' & _)]; [ destruct (Hbinds Hhu Hn) | ].
  exists vs'. intros g s_g Hg.
  split; [ exact (Hset g s_g (lookup_in _ _ _ Hg)) | ].
  exact (proj2 (Hbound' s slot syms decls vs' Hls') g s_g
           (lookup_in _ _ _ Hg)).
Qed.

(* [rebuild_exit_relation_whole] of a unit whose closures are made where
   the graph has only the facts of their sets and whose static values
   are of the fragment where it exits. *)
Lemma exit_relation_whole_of_exact_local (flag : reaper_preserve_direct_calls)
    (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool) :
  is_local_slots_only is_local ->
  in_flow_fragment e = true ->
  FlambdaReaperRebuildSpec.any_kind_arguments e = true ->
  FlambdaReaperRebuildSpec.results_kinded [] [] e = true ->
  well_formed_unit u e = true ->
  FlambdaReaperRebuildSpec.exits_one_value u e = true ->
  check_closed is_local (traverse_unit flag u e) T = true ->
  check_local_reads_used is_local (traverse_unit flag u e) T = true ->
  check_field_used_exact is_local (traverse_unit flag u e) T = true ->
  RP.closures_exact (graph_of_facts (traverse_unit flag u e)) (unit_kept e)
    e ->
  exits_of_values e ->
  rebuild_exit_relation_whole (graph_of_facts (traverse_unit flag u e))
    (solution_of_tables T) e
    (rebuild_unit_from_tables (traverse_unit flag u e) T
       preserve_direct_calls e).
Proof.
  intros Hlocal Hfrag Hdc Hrk Hwf Hexits Hcheck Hlrs Hcheck_exact Hxe Hvals
    n s k vs hp rd Hrun Hstep.
  split.
  - exact (proj1 (rebuild_exits_local flag is_local u e T
                    preserve_direct_calls n s k vs hp rd Hlocal Hfrag Hdc Hrk
                    Hwf Hexits Hcheck Hlrs Hcheck_exact Hrun Hstep)).
  - destruct (RP.rebuild_exit_statics_local flag is_local u e T
                preserve_direct_calls Hlocal Hfrag Hdc Hrk Hwf Hexits Hcheck
                Hlrs Hcheck_exact Hxe n s k vs hp rd Hrun Hstep)
      as (m & t & vs' & rd' & Hm & Ht & Hst & Hnl).
    exists m, t, vs', rd'.
    split; [ exact Hm | split; [ exact Ht | split ] ].
    + exact (statics_rel_rebuilt is_local _ _ _ _ _
               (check_closed_sound is_local _ _ Hcheck)
               (check_field_used_exact_sound is_local _ _ Hcheck_exact) Hst
               Hnl (Hvals n s k vs hp rd Hrun Hstep)).
    + apply (statics_rel_whole _ _ _ _ _ Hst).
      * exact (FlambdaClosuresProofs.well_formed_sets_bound u e n s Hwf Hrun).
      * exact (rebuilt_sets_bound _ T preserve_direct_calls u e m t Hwf Hm).
Qed.

Lemma exit_relation_whole_of_exact (flag : reaper_preserve_direct_calls)
    (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool) :
  is_local_slots_only is_local ->
  (∀ f : field, is_local f = false) ∨ projection_free e = true ->
  in_flow_fragment e = true ->
  FlambdaReaperRebuildSpec.any_kind_arguments e = true ->
  FlambdaReaperRebuildSpec.results_kinded [] [] e = true ->
  well_formed_unit u e = true ->
  FlambdaReaperRebuildSpec.exits_one_value u e = true ->
  check_closed is_local (traverse_unit flag u e) T = true ->
  check_field_used_exact is_local (traverse_unit flag u e) T = true ->
  RP.closures_exact (graph_of_facts (traverse_unit flag u e)) (unit_kept e)
    e ->
  exits_of_values e ->
  rebuild_exit_relation_whole (graph_of_facts (traverse_unit flag u e))
    (solution_of_tables T) e
    (rebuild_unit_from_tables (traverse_unit flag u e) T
       preserve_direct_calls e).
Proof.
  intros Hlocal Hslots Hfrag Hdc Hrk Hwf Hexits Hcheck Hcheck_exact.
  exact (exit_relation_whole_of_exact_local flag is_local u e T
           preserve_direct_calls Hlocal Hfrag Hdc Hrk Hwf Hexits Hcheck
           (FlambdaReaperFlowProofs.old_premise_local_reads_used flag
              is_local u e T Hlocal Hslots)
           Hcheck_exact).
Qed.

Theorem rebuild_exit_relation_whole_holds_local :
  rebuild_exit_relation_whole_holds_local_statement.
Proof.
  intros flag is_local u e T b Hlocal Hfrag Hdc Hrk Hwf Hexits Hcheck Hlrs
    Hcheck_exact Honce Hvals.
  apply (exit_relation_whole_of_exact_local flag is_local u e T b Hlocal Hfrag
           Hdc Hrk Hwf Hexits Hcheck Hlrs Hcheck_exact); [ | exact Hvals ].
  exact (FlambdaReaperClosuresExactProofs.closures_exact_of_bound_once flag u
           e Hfrag Hwf Honce).
Qed.

Theorem rebuild_exit_relation_whole_holds :
  rebuild_exit_relation_whole_holds_statement.
Proof.
  intros flag is_local u e T b Hlocal Hslots Hfrag Hdc Hrk Hwf Hexits Hcheck
    Hcheck_exact.
  exact (rebuild_exit_relation_whole_holds_local flag is_local u e T b Hlocal
           Hfrag Hdc Hrk Hwf Hexits Hcheck
           (FlambdaReaperFlowProofs.old_premise_local_reads_used flag
              is_local u e T Hlocal Hslots)
           Hcheck_exact).
Qed.

Theorem exported_unaliased_types_sound_end_to_end_local :
  exported_unaliased_types_sound_end_to_end_local_statement.
Proof.
  intros flag is_local u e T b sets tenv canon valuation valuation' n s k vs
    hp rd sym allocated Hlocal Hfrag Hdc Hrk Hwf Hexits Hleast Honce Hvals
    Hsets Hrun Hstep Henv Hfr Hcopies Hl.
  destruct (least_local_checks is_local _ T Hleast) as [Hcheck Hlrs].
  apply (exported_unaliased_types_sound_local flag is_local u e T b sets tenv
           canon valuation valuation' n s k vs hp rd sym allocated Hlocal Hleast
           Hsets);
    try assumption.
  exact (rebuild_exit_relation_whole_holds_local flag is_local u e T b Hlocal
           Hfrag Hdc Hrk Hwf Hexits Hcheck Hlrs
           (least_field_used_exact_local is_local _ T Hleast) Honce Hvals).
Qed.

Theorem exported_unaliased_types_end_to_end_local_landed :
  exported_unaliased_types_end_to_end_local_landed_statement.
Proof.
  intros flag is_local u e T b sets tenv canon valuation valuation' n s k vs
    hp rd sym allocated Hlocal Hfrag Hdc Hrk Hwf Hexits Hleast Honce Hvals
    Hsets Hrun Hstep Henv Hfr Hcopies Hl.
  destruct (least_local_checks is_local _ T Hleast) as [Hcheck Hlrs].
  apply (exported_unaliased_types_sound_local_landed flag is_local u e T b
           sets tenv canon valuation valuation' n s k vs hp rd sym allocated
           Hlocal Hleast Hsets);
    try assumption.
  exact (rebuild_exit_relation_whole_holds_local flag is_local u e T b Hlocal
           Hfrag Hdc Hrk Hwf Hexits Hcheck Hlrs
           (least_field_used_exact_local is_local _ T Hleast) Honce Hvals).
Qed.

Theorem exported_unaliased_types_sound_end_to_end :
  exported_unaliased_types_sound_end_to_end_statement.
Proof.
  intros flag is_local u e T b sets tenv canon valuation valuation' n s k vs
    hp rd sym allocated Hlocal Hslots Hfrag Hdc Hrk Hwf Hexits Hleast.
  exact (exported_unaliased_types_sound_end_to_end_local flag is_local u e T b
           sets tenv canon valuation valuation' n s k vs hp rd sym allocated
           Hlocal Hfrag Hdc Hrk Hwf Hexits
           (least_local_of_least is_local _ T Hleast
              (FlambdaReaperFlowProofs.old_premise_local_reads_used flag
                 is_local u e T Hlocal Hslots))).
Qed.

Theorem exported_unaliased_types_sound_end_to_end_landed :
  exported_unaliased_types_sound_end_to_end_landed_statement.
Proof.
  intros flag is_local u e T b sets tenv canon valuation valuation' n s k vs
    hp rd sym allocated Hlocal Hslots Hfrag Hdc Hrk Hwf Hexits Hleast.
  exact (exported_unaliased_types_end_to_end_local_landed flag is_local u e T b
           sets tenv canon valuation valuation' n s k vs hp rd sym allocated
           Hlocal Hfrag Hdc Hrk Hwf Hexits
           (least_local_of_least is_local _ T Hleast
              (FlambdaReaperFlowProofs.old_premise_local_reads_used flag
                 is_local u e T Hlocal Hslots))).
Qed.

(* ======================================================================== *)
(* Closures whose representation changes                                    *)
(* ======================================================================== *)

From OxCamlFlambda2 Require FlambdaReaperUnboxingRefinesProofs
  FlambdaReaperUnboxingRefinesClosuresSimProofs FlambdaReaperUnboxingProofs.

(* A membership of a list computed, as the chain of its equalities. *)
Ltac in_cases H :=
  repeat match type of H with
         | _ ∨ _ => destruct H as [H | H]
         end;
  try discriminate H; try contradiction H.

Ltac in_find :=
  repeat first [ left; reflexivity | right ].

(* The constructors of the graph of the example. *)
Lemma changed_example_constructors :
  f_constructor FlambdaReaperUnboxing.Examples.block_in_slot_facts
  = [(Node_variable 60, Field_call_witness Known_arity_code_pointer,
      Node_made 7);
     (Node_variable 60, Field_call_witness Unknown_arity_code_pointer,
      Node_made 8);
     (Node_variable 60, Field_value_slot 80, Node_variable 10);
     (Node_variable 60, Field_function_slot 70, Node_variable 60);
     (Node_variable 10, Field_block 0 Value, Node_all_constants);
     (Node_variable 10, Field_block 1 Value, Node_all_constants);
     (Node_variable 10, Field_is_int, Node_all_constants);
     (Node_variable 10, Field_get_tag, Node_all_constants);
     (Node_made 8, Field_return_of_call Exn, Node_made 4);
     (Node_made 8, Field_return_of_call (Normal 0), Node_made 3);
     (Node_made 8, Field_code_id_of_call_witness, Node_code_id 52);
     (Node_made 7, Field_return_of_call (Normal 0), Node_made 3);
     (Node_made 7, Field_return_of_call Exn, Node_made 4);
     (Node_made 7, Field_code_id_of_call_witness, Node_code_id 52)]%positive.
Proof. vm_compute. reflexivity. Qed.

(* A constructor of a graph of facts, without unfolding the facts: the
   kernel would compute them by lazy reduction. *)
Lemma g_constructor_facts (F : facts) (a : node) (f : field) (b : node) :
  g_constructor (graph_of_facts F) a f b -> (a, f, b) ∈ f_constructor F.
Proof. exact (fun H => H). Qed.

Theorem changed_example : changed_example_statement.
Proof.
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ apply type_satisfied_b_iff; vm_compute; reflexivity | ].
  split; [ | split ].
  - right. exists ChangedExample.x60. split; [ | split ].
    + cbn [rebuilt_value ChangedExample.closure].
      exists (fun _ => ChangedExample.x60),
        [(80%positive,
          Value_block 0
            [Value_const (Tagged_immediate 7); Value_const (Poison Value)])].
      split; [ reflexivity | split ].
      { apply may_come_from_self; vm_compute; auto. }
      split.
      * intros f id Hf g b. cbn [lookup] in Hf.
        destruct (Pos.eqb_spec f 70) as [-> | _]; [ | discriminate Hf ].
        split.
        -- intros Hcons.
           apply g_constructor_facts in Hcons.
           rewrite changed_example_constructors in Hcons.
           cbn [In] in Hcons. in_cases Hcons.
           injection Hcons as <- <-.
           split; [ reflexivity | reflexivity ].
        -- intros [Hg ->]. unfold is_bound in Hg. cbn [lookup] in Hg.
           destruct (Pos.eqb g 70) eqn:Hg70; [ | discriminate Hg ].
           apply Pos.eqb_eq in Hg70. subst g. vm_compute. in_find.
      * apply for_all_bindings_lookup. intros w x Hw f id Hf Hfu.
        cbn [lookup] in Hw.
        destruct (Pos.eqb_spec w 80) as [-> | _]; [ | discriminate Hw ].
        injection Hw as <-.
        exists (Node_variable 10%positive),
          (Value_block 0
             [Value_const (Tagged_immediate 7); Value_const (Poison Value)]).
        split; [ | split; [ vm_compute; reflexivity | ] ].
        { intros f' id' _ b. split.
          - intros Hcons.
            apply g_constructor_facts in Hcons.
            rewrite changed_example_constructors in Hcons.
            cbn [In] in Hcons. in_cases Hcons.
            injection Hcons as <-. reflexivity.
          - intros ->. vm_compute. in_find. }
        split; [ vm_compute; reflexivity | split; [ reflexivity | ] ].
        cbn [rebuilt_value].
        exists (Node_variable 10%positive),
          [Value_const (Tagged_immediate 7); Value_const (Poison Value)].
        split; [ apply may_come_from_self; vm_compute; auto | ].
        split; [ vm_compute; reflexivity | split; [ reflexivity | ] ].
        cbn [fields_rebuilt kind_of_value kind_of_const].
        split; [ | split; [ | exact I ] ].
        -- exists Node_all_constants. split; [ vm_compute; in_find | ].
           split; [ | intros H; vm_compute in H; discriminate H ].
           intros _. split; [ vm_compute; reflexivity | ].
           cbn [rebuilt_value].
           split; [ vm_compute; reflexivity | reflexivity ].
        -- exists Node_all_constants. split; [ vm_compute; in_find | ].
           split; [ intros H; vm_compute in H; discriminate H | ].
           intros _. reflexivity.
    + intros a (_ & [Has | Hsrc] & _) _;
        [ vm_compute in Has; discriminate Has | ].
      cbn [s_sources solution_of_tables] in Hsrc.
      apply mem_pair_In in Hsrc. vm_compute in Hsrc. in_cases Hsrc.
      injection Hsrc as <-. reflexivity.
    + intros _. vm_compute. reflexivity.
  - exists [(70%positive, 52%positive)],
      [(80%positive,
        Value_block 0
          [Value_const (Tagged_immediate 7); Value_const (Poison Value)])],
      [(81%positive, Value_const (Tagged_immediate 7))].
    split; [ reflexivity | ].
    split; [ vm_compute; reflexivity | ].
    split; [ vm_compute; reflexivity | ].
    split; [ vm_compute; reflexivity | ].
    vm_compute. constructor; [ split; reflexivity | constructor ].
  - apply type_satisfied_b_iff. vm_compute. reflexivity.
Qed.

Theorem changed_example_landed : changed_example_landed_statement.
Proof.
  destruct changed_example as (Hreprs & _ & _ & Hrest).
  split; [ exact Hreprs | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  exact Hrest.
Qed.

(* The closures of the example are related as the simulation of the
   rebuild with unboxing relates them
   (FlambdaReaperUnboxingRefinesClosuresSimProofs.crel). *)
Example changed_example_crel :
  FlambdaReaperUnboxingRefinesClosuresSimProofs.crel
    FlambdaReaperUnboxing.Examples.block_in_slot_repr
    ChangedExample.closure_mid ChangedExample.closure'.
Proof.
  exists [(70%positive, 52%positive)],
    [(80%positive,
      Value_block 0
        [Value_const (Tagged_immediate 7); Value_const (Poison Value)])],
    [(81%positive, Value_const (Tagged_immediate 7))].
  split; [ reflexivity | split; [ vm_compute; reflexivity | ] ].
  split; [ vm_compute; reflexivity | ].
  intros w u Hin. cbn in Hin. destruct Hin as [Hin | []].
  injection Hin as <- <-. eexists. split; [ reflexivity | ].
  cbn [FlambdaReaperUnboxingRefinesClosuresSimProofs.slot_holds].
  intros p x k Hp.
  inversion Hp as [ | k0 fields f u0 p0 x0 k1 Hfu Hsub ]; subst.
  destruct Hfu as [Hfu | []]. injection Hfu as <- <-.
  inversion Hsub; subst.
  eexists. split; [ reflexivity | split; reflexivity ].
Qed.

(* ======================================================================== *)
(* The leaves of a representation                                           *)
(* ======================================================================== *)

(* The values of the slots of a map whose keys are those of another, at
   the same places. *)
Lemma combine_lookup {A : Type} (P : value -> flambda_type -> Prop)
    (g : A -> flambda_type) :
  ∀ (L : list A) (FL : list (positive * flambda_kind))
    (vs' : list (positive * value)),
    length L = length FL -> map fst vs' = map fst FL ->
    Forall2 (fun l x' => P x' (g l)) L (map snd vs') ->
    map fst (map (fun q => (fst (snd q), g (fst q))) (combine L FL))
    = map fst vs'
    ∧ ∀ (w : positive) (ty : flambda_type) (x' : value),
        lookup w (map (fun q => (fst (snd q), g (fst q))) (combine L FL))
        = Some ty ->
        lookup w vs' = Some x' -> P x' ty.
Proof.
  induction L as [ | l L IH]; intros [ | [s k] FL] [ | [s' x] vs'] Hlen Hkeys
    HF; cbn [length map fst snd combine] in *; try discriminate;
    try (inversion HF; fail).
  - split; [ reflexivity | discriminate ].
  - injection Hkeys as -> Hkeys. injection Hlen as Hlen.
    inversion HF as [ | l0 x0 L0 xs0 Hx HF' ]; subst.
    destruct (IH FL vs' Hlen Hkeys HF') as [IH1 IH2].
    split; [ rewrite IH1; reflexivity | ].
    intros w ty x' Hw Hx'. cbn [lookup] in Hw, Hx'.
    destruct (Pos.eqb w s).
    + injection Hw as <-. injection Hx' as <-. exact Hx.
    + exact (IH2 w ty x' Hw Hx').
Qed.

(* The classification of fields read is not empty. *)
Lemma classify_step_not_empty (acc : field_class) (f : field) :
  classify_step acc f <> Class_empty.
Proof.
  destruct acc; destruct f; cbn; try discriminate;
    match goal with
    | |- (if ?b then _ else _) <> _ => destruct b; discriminate
    end.
Qed.

Lemma classify_fold_not_empty (l : list field) :
  ∀ acc : field_class,
    acc <> Class_empty -> List.fold_left classify_step l acc <> Class_empty.
Proof.
  induction l as [ | f l IH]; intros acc Hacc; [ exact Hacc | ].
  apply IH. apply classify_step_not_empty.
Qed.

Lemma classify_cons_not_empty (f : field) (l : list field) :
  classify_field_map (f :: l) <> Class_empty.
Proof.
  unfold classify_field_map. cbn [List.fold_left].
  apply classify_fold_not_empty, classify_step_not_empty.
Qed.

(* The type of a value slot of a closure type. *)
Lemma destructure_value_slot (e : Expanded_type.t)
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
  destructure_field (Field_value_slot w) (Some e)
  = match lookup w vsts with
    | Some ty => ty
    | None => unknown Value
    end.
Proof.
  intros He Hs. unfold destructure_field. rewrite He.
  cbn [destructure_non_null]. unfold row_like_for_closures_get_env_var.
  destruct (row_like_for_closures_get_single_tag (Row_like_for_closures known))
    as [[ | s en | s en] | ]; cbn in Hs; try discriminate;
    injection Hs as _ Hen; subst en; destruct (lookup w vsts); reflexivity.
Qed.

(* The type at a path below an unknown type, where a value is. *)
Lemma destructure_path_unknown (tenv : Typing_env.t) :
  ∀ (u : U.unboxed_field) (kind : flambda_kind) (p : list field)
    (k : flambda_kind) (y x : value),
    (p, k) ∈ unboxed_field_paths kind u -> value_at_path p y = Some x ->
    destructure_path tenv p (unknown kind) = unknown k.
Proof.
  refine (URP.unboxed_field_rec _ _ _).
  - intros v kind p k y x Hin _. destruct Hin as [Heq | []].
    injection Heq as <- <-. reflexivity.
  - intros fields IH kind p k y x Hin Hv.
    cbn [unboxed_field_paths] in Hin.
    induction fields as [ | [f u] fields IHl]; [ destruct Hin | ].
    apply in_app_or in Hin as [Hin | Hin].
    + apply in_map_iff in Hin as ([p' k'] & Heq & Hin).
      injection Heq as <- <-. cbn [fst snd] in *.
      cbn [value_at_path] in Hv.
      destruct (field_of_value f y) as [y1 | ] eqn:Hfv; [ | discriminate Hv ].
      cbn [destructure_path]. rewrite expand_unknown.
      assert (Hua : destructure_field f
                      (Some (Expanded_type.create_unknown kind))
                    = unknown (U.field_kind f)).
      { destruct kind as [ | [] | | ];
          destruct f; cbn [field_of_value] in Hfv; try discriminate Hfv;
          reflexivity. }
      rewrite Hua.
      exact (IH f u (or_introl eq_refl) (U.field_kind f) p' k' y1 x Hin Hv).
    + exact (IHl (fun f' u' H => IH f' u' (or_intror H)) Hin).
Qed.

(* A leaf, with the membership of its path. *)
Lemma Forall2_strip {B : Type} (R : list field * flambda_kind -> B -> Prop) :
  ∀ (L : list unboxed_leaf) (xs : list B),
    Forall2 R (map (fun l => let '(p, k, _) := l in (p, k)) L) xs ->
    Forall2
      (fun l x =>
         (let '(p, k, _) := l in (p, k))
           ∈ map (fun l => let '(p, k, _) := l in (p, k)) L
         ∧ R (let '(p, k, _) := l in (p, k)) x)
      L xs.
Proof.
  intros L xs H.
  assert (Hgen : ∀ L0 : list unboxed_leaf,
            (∀ l, l ∈ L0 -> l ∈ L) ->
            Forall2 R (map (fun l => let '(p, k, _) := l in (p, k)) L0) xs ->
            Forall2
              (fun l x =>
                 (let '(p, k, _) := l in (p, k))
                   ∈ map (fun l => let '(p, k, _) := l in (p, k)) L
                 ∧ R (let '(p, k, _) := l in (p, k)) x)
              L0 xs).
  { clear H. intros L0. revert xs.
    induction L0 as [ | l L0 IH]; intros xs Hsub H; inversion H; subst;
      constructor.
    - split; [ | assumption ].
      apply (in_map (fun l : unboxed_leaf => let '(p, k, _) := l in (p, k))).
      apply Hsub. left. reflexivity.
    - apply IH; [ intros l1 Hl1; apply Hsub; right; exact Hl1 | assumption ].
  }
  exact (Hgen L (fun l H => H) H).
Qed.

(* ======================================================================== *)
(* The fields of the closures of a set ([get_fields_of_constructors])        *)
(* ======================================================================== *)

Lemma In_dedup_fields (x : field) (l : list field) :
  x ∈ Keyed.dedup field_eqb l <-> x ∈ l.
Proof.
  rewrite <- !(keyed_among_in field field_eqb URP.field_eqb_decides).
  rewrite (keyed_among_dedup field field_eqb URP.field_eqb_decides).
  reflexivity.
Qed.

Lemma field_eqb_sym (a b : field) : field_eqb a b = field_eqb b a.
Proof.
  destruct (field_eqb a b) eqn:E; destruct (field_eqb b a) eqn:E';
    try reflexivity.
  - apply field_eqb_eq in E. subst. rewrite field_eqb_refl in E'.
    discriminate E'.
  - apply field_eqb_eq in E'. subst. rewrite field_eqb_refl in E.
    discriminate E.
Qed.

Section Constructor_fields.

Variable is_local : field -> bool.
Variable F : facts.
Variable T : tables.
Variable sets : list set_of_closures.
Hypothesis Hleast : least_either is_local F T.

Local Abbreviation ctx := (plain_context is_local F T sets).
Local Abbreviation G := (graph_of_facts F).
Local Abbreviation sol := (solution_of_tables T).

(* The queries of FlambdaReaperUnboxing.v, read back. *)
Lemma has_constructor_agree (c : node) (f : field) :
  U.has_constructor F c f = negb (is_nil (constructors_of ctx c f)).
Proof.
  apply Bool.eq_iff_eq_true. unfold U.has_constructor. rewrite existsb_exists.
  split.
  - intros ([[b f'] from] & Hin & Hb). apply andb_true_iff in Hb as [Hb Hf].
    apply node_eqb_eq in Hb. apply field_eqb_eq in Hf. subst.
    assert (Hfrom : from ∈ constructors_of ctx b f')
      by (apply constructors_of_iff; exact Hin).
    destruct (constructors_of ctx b f'); [ destruct Hfrom | reflexivity ].
  - intros H. destruct (constructors_of ctx c f) as [ | from l] eqn:E;
      [ discriminate H | ].
    assert (Hfrom : from ∈ constructors_of ctx c f)
      by (rewrite E; left; reflexivity).
    apply constructors_of_iff in Hfrom.
    exists (c, f, from). split; [ exact Hfrom | ].
    cbn. rewrite node_eqb_refl, field_eqb_refl. reflexivity.
Qed.

Lemma uses_of_iff (c u : node) :
  u ∈ U.uses_of T c <-> s_usages sol c u = true.
Proof.
  unfold U.uses_of. rewrite in_flat_map. cbn [solution_of_tables s_usages].
  rewrite mem_pair_In. split.
  - intros ([x y] & Hin & Hu). destruct (node_eqb x c) eqn:E;
      [ | destruct Hu ].
    destruct Hu as [<- | []]. apply node_eqb_eq in E. subst. exact Hin.
  - intros Hin. exists (c, u). split; [ exact Hin | ].
    cbn. rewrite node_eqb_refl. left. reflexivity.
Qed.

Lemma sourced_by_iff (c u : node) :
  u ∈ U.sourced_by T c <-> s_sources sol u c = true.
Proof.
  unfold U.sourced_by. rewrite in_flat_map. cbn [solution_of_tables s_sources].
  rewrite mem_pair_In. split.
  - intros ([x y] & Hin & Hu). destruct (node_eqb y c) eqn:E;
      [ | destruct Hu ].
    destruct Hu as [<- | []]. apply node_eqb_eq in E. subst. exact Hin.
  - intros Hin. exists (u, c). split; [ exact Hin | ].
    cbn. rewrite node_eqb_refl. left. reflexivity.
Qed.

Lemma types_sourced_by_iff (c u : node) :
  u ∈ sourced_by ctx c <-> s_sources sol u c = true.
Proof.
  unfold sourced_by. rewrite In_filter_map_fst.
  cbn [solution_of_tables s_sources]. rewrite mem_pair_In. reflexivity.
Qed.

Lemma read_into_any_usage_iff (u : node) (f : field) :
  U.read_into_any_usage F T u f = true
  <-> ∃ y : node, g_accessor G y f u ∧ s_any_usage sol y = true.
Proof.
  unfold U.read_into_any_usage. rewrite existsb_exists. split.
  - intros ([[v f'] u'] & Hin & H). apply andb_true_iff in H as [H Hau].
    apply andb_true_iff in H as [Hu Hf]. apply node_eqb_eq in Hu.
    apply field_eqb_eq in Hf. subst. exists v. split; [ exact Hin | ].
    exact Hau.
  - intros (y & Hin & Hau). exists (y, f, u). split; [ exact Hin | ].
    cbn. rewrite node_eqb_refl, field_eqb_refl. exact Hau.
Qed.

Lemma read_into_iff (u y : node) (f : field) :
  y ∈ U.read_into F T u f
  <-> g_accessor G y f u ∧ s_has_usage sol y = true.
Proof.
  unfold U.read_into. rewrite in_flat_map. split.
  - intros ([[v f'] u'] & Hin & H).
    destruct (node_eqb u u' && field_eqb f f' && mem_node v (t_has_usage T))
      eqn:E; [ | destruct H ].
    destruct H as [<- | []].
    apply andb_true_iff in E as [E Hhu]. apply andb_true_iff in E as [Hu Hf].
    apply node_eqb_eq in Hu. apply field_eqb_eq in Hf. subst.
    split; [ exact Hin | exact Hhu ].
  - intros [Hin Hhu]. exists (y, f, u). split; [ exact Hin | ].
    cbn. rewrite node_eqb_refl, field_eqb_refl.
    cbn [solution_of_tables s_has_usage] in Hhu. rewrite Hhu. left.
    reflexivity.
Qed.

Lemma existsb_accessors_iff (u : node) (f : field) :
  existsb (any_usage ctx) (accessors_of ctx f u) = true
  <-> ∃ y : node, g_accessor G y f u ∧ s_any_usage sol y = true.
Proof.
  rewrite existsb_exists. split.
  - intros (y & Hy & Hau). apply accessors_of_iff in Hy. exists y. auto.
  - intros (y & Hy & Hau). exists y.
    split; [ apply accessors_of_iff; exact Hy | exact Hau ].
Qed.

Lemma field_used_top_agree (c : node) (f : field) :
  U.field_used_top F T is_local c f = field_used_top ctx c f.
Proof.
  assert (E1 : ∀ l l' : list node,
             (∀ u, u ∈ l <-> u ∈ l') ->
             existsb (fun usage => U.read_into_any_usage F T usage f) l
             = existsb (fun u => existsb (any_usage ctx) (accessors_of ctx f u))
                 l').
  { intros l l' Hl. apply Bool.eq_iff_eq_true. rewrite !existsb_exists.
    split.
    - intros (u & Hu & H). apply Hl in Hu.
      apply read_into_any_usage_iff in H.
      exists u. split; [ exact Hu | apply existsb_accessors_iff; exact H ].
    - intros (u & Hu & H). apply Hl in Hu.
      apply existsb_accessors_iff in H.
      exists u. split; [ exact Hu | apply read_into_any_usage_iff; exact H ].
  }
  unfold U.field_used_top, field_used_top. rewrite has_constructor_agree.
  rewrite (E1 (U.uses_of T c) (usages_of ctx c)).
  2: { intros u. rewrite uses_of_iff, usages_of_iff. reflexivity. }
  rewrite (E1 (U.sourced_by T c) (sourced_by ctx c)).
  2: { intros u. rewrite sourced_by_iff, types_sourced_by_iff. reflexivity. }
  assert (E2 : ∀ l : list (field * node),
             existsb (fun '(f', x) => field_eqb f f' && mem_node x
                                        (t_any_usage T)) l
             = existsb (fun p => field_eqb (fst p) f
                                 && any_usage ctx (snd p)) l).
  { induction l as [ | [f' x] l IH]; [ reflexivity | ].
    cbn [existsb fst snd]. rewrite IH, field_eqb_sym. reflexivity. }
  rewrite E2. reflexivity.
Qed.

Lemma field_used_as_agree (c y : node) (f : field) :
  y ∈ U.field_used_as F T is_local c f <-> y ∈ field_used_as ctx c f.
Proof.
  assert (E1 : ∀ l l' : list node,
             (∀ u, u ∈ l <-> u ∈ l') ->
             y ∈ flat_map (fun usage => U.read_into F T usage f) l
             <-> y ∈ flat_map (fun u => filter (has_use ctx)
                                          (accessors_of ctx f u)) l').
  { intros l l' Hl. rewrite !in_flat_map. split.
    - intros (u & Hu & H). apply Hl in Hu. apply read_into_iff in H as [H1 H2].
      exists u. split; [ exact Hu | ]. apply In_filter_has_use.
      split; [ apply accessors_of_iff; exact H1 | exact H2 ].
    - intros (u & Hu & H). apply Hl in Hu.
      apply In_filter_has_use in H as [H1 H2]. apply accessors_of_iff in H1.
      exists u. split; [ exact Hu | apply read_into_iff; auto ]. }
  unfold U.field_used_as, field_used_as. rewrite has_constructor_agree.
  destruct (is_nil (constructors_of ctx c f)); cbn [negb]; [ reflexivity | ].
  rewrite In_dedup_nodes, !in_app_iff.
  rewrite (E1 (U.uses_of T c) (usages_of ctx c)).
  2: { intros u. rewrite uses_of_iff, usages_of_iff. reflexivity. }
  cbn [plain_context rc_is_local]. unfold any_usage.
  cbn [plain_context rc_tables].
  destruct (mem_node c (t_any_usage T));
    [ rewrite andb_true_r | rewrite andb_false_r; reflexivity ].
  assert (E3 : y ∈ (if is_local f
                    then flat_map (fun usage => U.read_into F T usage f)
                           (U.sourced_by T c)
                    else [])
               <-> y ∈ (if is_local f
                        then flat_map (fun u => filter (has_use ctx)
                                                  (accessors_of ctx f u))
                               (sourced_by ctx c)
                        else [])).
  { destruct (is_local f); [ | reflexivity ].
    apply E1. intros u. rewrite sourced_by_iff, types_sourced_by_iff.
    reflexivity. }
  rewrite E3. apply or_iff_compat_l. apply or_iff_compat_r.
  rewrite in_flat_map, in_map_iff. split.
  - intros ([f' x] & Hin & H).
    destruct (field_eqb f f' && mem_node x (t_has_usage T)) eqn:E;
      [ | destruct H ].
    destruct H as [-> | []]. exists (f', y). split; [ reflexivity | ].
    apply filter_In. split; [ exact Hin | ]. cbn. rewrite field_eqb_sym.
    exact E.
  - intros ([f' x] & <- & Hin). apply filter_In in Hin as [Hin E].
    exists (f', x). split; [ exact Hin | ]. cbn in E |- *.
    rewrite field_eqb_sym in E. unfold has_use in E. cbn in E.
    rewrite E. left. reflexivity.
Qed.

(* A field of [field_used_top] or [field_used_as] is used. *)
Lemma field_used_top_used (c : node) (f : field) :
  field_used_top ctx c f = true -> s_field_used sol c f = true.
Proof.
  intros Htop. pose proof (least_closed_tables is_local F T Hleast) as Hc.
  unfold field_used_top in Htop. cbn [plain_context rc_is_local] in Htop.
  apply andb_true_iff in Htop as [Hne Htop].
  destruct (constructors_of ctx c f) as [ | from l] eqn:E;
    [ discriminate Hne | ].
  assert (Hcons : g_constructor G c f from).
  { apply (constructors_of_iff ctx). rewrite E. left. reflexivity. }
  repeat apply orb_true_iff in Htop as [Htop | Htop].
  - apply andb_true_iff in Htop as [Hau Hnl]. apply negb_true_iff in Hnl.
    exact (field_used_any_usage _ _ _ Hc c f from Hcons Hau Hnl).
  - apply existsb_exists in Htop as (u & Hu & Hy).
    apply existsb_accessors_iff in Hy as (y & Hy & Hauy).
    apply usages_of_iff in Hu.
    exact (field_used_usage _ _ _ Hc c f from u y Hcons Hu Hy
             (has_usage_any _ _ _ Hc y Hauy)).
  - apply andb_true_iff in Htop as [Hau Hread].
    apply existsb_exists in Hread as ([f' x] & Hin & Hx).
    apply andb_true_iff in Hx as [Hf Haux]. cbn [fst snd] in Hf, Haux.
    apply field_eqb_eq in Hf. subst f'.
    assert (Hrf : s_reading_field sol f x = true).
    { cbn [solution_of_tables s_reading_field].
      apply mem_field_node_In. exact Hin. }
    exact (field_used_reading _ _ _ Hc c f from x Hcons Hau Hrf
             (has_usage_any _ _ _ Hc x Haux)).
  - apply andb_true_iff in Htop as [Htop Hsrc].
    apply andb_true_iff in Htop as [Hloc Hau].
    apply existsb_exists in Hsrc as (u & Hu & Hy).
    apply existsb_accessors_iff in Hy as (y & Hy & Hauy).
    apply types_sourced_by_iff in Hu.
    exact (field_used_local_source _ _ _ Hc u c f from y Hcons Hu Hloc Hau Hy
             (has_usage_any _ _ _ Hc y Hauy)).
Qed.

Lemma field_used_as_used (c y : node) (f : field) :
  y ∈ field_used_as ctx c f -> s_field_used sol c f = true.
Proof.
  intros Hy. pose proof (least_closed_tables is_local F T Hleast) as Hc.
  unfold field_used_as in Hy.
  destruct (constructors_of ctx c f) as [ | from l] eqn:E; [ destruct Hy | ].
  assert (Hcons : g_constructor G c f from).
  { apply (constructors_of_iff ctx). rewrite E. left. reflexivity. }
  cbn [is_nil] in Hy. apply (proj1 (In_dedup_nodes y _)) in Hy.
  apply in_app_or in Hy as [Hy | Hy]; [ | apply in_app_or in Hy as [Hy | Hy] ].
  - apply in_flat_map in Hy as (u & Hu & Hy).
    apply In_filter_has_use in Hy as [Hy Hhu].
    apply usages_of_iff in Hu. apply accessors_of_iff in Hy.
    exact (field_used_usage _ _ _ Hc c f from u y Hcons Hu Hy Hhu).
  - destruct (any_usage ctx c) eqn:Hau; [ | destruct Hy ].
    apply in_map_iff in Hy as ([f' y'] & Hy' & Hin). cbn [snd] in Hy'.
    subst y'. apply filter_In in Hin as [Hin Hfy].
    apply andb_true_iff in Hfy as [Hf Hhu]. cbn [fst snd] in Hf, Hhu.
    apply field_eqb_eq in Hf. subst f'.
    assert (Hrf : s_reading_field sol f y = true).
    { cbn [solution_of_tables s_reading_field].
      apply mem_field_node_In. exact Hin. }
    exact (field_used_reading _ _ _ Hc c f from y Hcons Hau Hrf Hhu).
  - destruct (rc_is_local ctx f && any_usage ctx c) eqn:Hla;
      [ | destruct Hy ].
    apply andb_true_iff in Hla as [Hloc Hau].
    apply in_flat_map in Hy as (u & Hu & Hy).
    apply In_filter_has_use in Hy as [Hy Hhu].
    apply types_sourced_by_iff in Hu. apply accessors_of_iff in Hy.
    exact (field_used_local_source _ _ _ Hc u c f from y Hcons Hu Hloc Hau Hy
             Hhu).
Qed.

(* [get_fields_of_constructors] of the closures [cl]. *)
Lemma gfoc_spec (cl : list node) (f : field) (use : U.field_use) :
  (f, use) ∈ U.get_fields_of_constructors F T is_local cl ->
  is_function_slot f = false
  ∧ match use with
    | U.Use_unknown => ∃ c : node, c ∈ cl ∧ field_used_top ctx c f = true
    | U.Use_known flow_to =>
      flow_to <> []
      ∧ (∀ c : node, c ∈ cl -> field_used_top ctx c f = false)
      ∧ ∀ y : node,
          y ∈ flow_to <-> ∃ c : node, c ∈ cl ∧ y ∈ field_used_as ctx c f
    end.
Proof.
  unfold U.get_fields_of_constructors. intros Hin.
  apply URP.sort_fields_in, in_flat_map in Hin as (f0 & Hf0 & Hin).
  apply (proj1 (In_dedup_fields _ _)), in_flat_map in Hf0
    as ([[b f1] x] & _ & Hf1).
  destruct (mem_node b cl && negb (is_function_slot f1)) eqn:Hb;
    [ | destruct Hf1 ].
  destruct Hf1 as [<- | []]. apply andb_true_iff in Hb as [_ Hfs].
  apply negb_true_iff in Hfs.
  destruct (existsb (fun x => U.field_used_top F T is_local x f1) cl)
    eqn:Htop.
  - destruct Hin as [Heq | []]. injection Heq as <- <-.
    split; [ exact Hfs | ].
    apply existsb_exists in Htop as (c & Hc & Htop). exists c.
    rewrite <- field_used_top_agree. auto.
  - destruct (Keyed.dedup node_eqb
                (flat_map (fun x => U.field_used_as F T is_local x f1) cl))
      as [ | y0 ys] eqn:Hl; [ destruct Hin | ].
    destruct Hin as [Heq | []]. injection Heq as <- <-.
    split; [ exact Hfs | split; [ discriminate | split ] ].
    + intros c Hc. rewrite <- field_used_top_agree.
      apply not_true_iff_false. intros H. apply not_true_iff_false in Htop.
      apply Htop. apply existsb_exists. exists c. auto.
    + intros y. rewrite <- Hl. split.
      * intros Hy. apply URP.dedup_in, in_flat_map in Hy as (c & Hc & Hy).
        exists c. split; [ exact Hc | apply field_used_as_agree; exact Hy ].
      * intros (c & Hc & Hy). apply URP.dedup_in_r, in_flat_map.
        exists c. split; [ exact Hc | apply field_used_as_agree; exact Hy ].
Qed.

(* A field of it is used by a closure of [cl]. *)
Lemma gfoc_used (cl : list node) (f : field) (use : U.field_use) :
  (f, use) ∈ U.get_fields_of_constructors F T is_local cl ->
  ∃ c : node, c ∈ cl ∧ s_field_used sol c f = true.
Proof.
  intros Hin. destruct (gfoc_spec cl f use Hin) as [_ Hu].
  destruct use as [ | flow_to].
  - destruct Hu as (c & Hc & Htop). exists c.
    split; [ exact Hc | exact (field_used_top_used c f Htop) ].
  - destruct Hu as (Hne & _ & Hys).
    destruct flow_to as [ | y ys]; [ contradiction | ].
    destruct (proj1 (Hys y) (or_introl eq_refl)) as (c & Hc & Hy).
    exists c. split; [ exact Hc | exact (field_used_as_used c y f Hy) ].
Qed.

(* It reads the closures given as a set. *)
Lemma gfoc_ext (cl cl' : list node) (f : field) (use : U.field_use) :
  (∀ c : node, c ∈ cl <-> c ∈ cl') ->
  (f, use) ∈ U.get_fields_of_constructors F T is_local cl ->
  ∃ use' : U.field_use,
    (f, use') ∈ U.get_fields_of_constructors F T is_local cl'.
Proof.
  intros Hcl Hin.
  assert (Hmem : ∀ b, mem_node b cl = mem_node b cl').
  { intros b. apply Bool.eq_iff_eq_true. rewrite !mem_node_In. apply Hcl. }
  unfold U.get_fields_of_constructors in Hin |- *.
  apply URP.sort_fields_in, in_flat_map in Hin as (f0 & Hf0 & Hin).
  assert (Hkeys : ∀ t : node * field * node,
            (let '(base, f, _) := t in
             if mem_node base cl && negb (is_function_slot f) then [f]
             else [])
            = (let '(base, f, _) := t in
               if mem_node base cl' && negb (is_function_slot f) then [f]
               else [])).
  { intros [[b f1] x]. rewrite Hmem. reflexivity. }
  rewrite (flat_map_ext _ _ Hkeys) in Hf0.
  assert (Htop : existsb (fun x => U.field_used_top F T is_local x f0) cl
                 = existsb (fun x => U.field_used_top F T is_local x f0) cl').
  { apply Bool.eq_iff_eq_true. rewrite !existsb_exists.
    split; intros (c & Hc & H); exists c; split; auto; apply Hcl; exact Hc. }
  destruct (existsb (fun x => U.field_used_top F T is_local x f0) cl)
    eqn:Ht.
  - destruct Hin as [Heq | []]. injection Heq as -> <-.
    exists U.Use_unknown. apply URP.sort_fields_in_r, in_flat_map.
    exists f. split; [ exact Hf0 | ]. rewrite <- Htop. left. reflexivity.
  - destruct (Keyed.dedup node_eqb
                (flat_map (fun x => U.field_used_as F T is_local x f0) cl))
      as [ | y0 ys] eqn:Hl; [ destruct Hin | ].
    destruct Hin as [Heq | []]. injection Heq as -> <-.
    assert (Hy0 : y0 ∈ flat_map (fun x => U.field_used_as F T is_local x f)
                         cl').
    { assert (H0 : y0 ∈ Keyed.dedup node_eqb
                          (flat_map (fun x => U.field_used_as F T is_local x f)
                             cl))
        by (rewrite Hl; left; reflexivity).
      apply URP.dedup_in, in_flat_map in H0 as (c & Hc & H0).
      apply in_flat_map. exists c. split; [ apply Hcl; exact Hc | exact H0 ]. }
    destruct (Keyed.dedup node_eqb
                (flat_map (fun x => U.field_used_as F T is_local x f) cl'))
      as [ | y1 ys'] eqn:Hl'.
    { apply URP.dedup_in_r in Hy0. rewrite Hl' in Hy0. destruct Hy0. }
    exists (U.Use_known (y1 :: ys')). apply URP.sort_fields_in_r, in_flat_map.
    exists f. split; [ exact Hf0 | ]. rewrite <- Htop, Hl'. left. reflexivity.
Qed.

(* The metadata of a leaf of a value slot of the set holds where the
   closures of [cl] hold the slot. *)
Lemma gfoc_metadata_at (cl : list node) (a from : node) (f : field)
    (use : U.field_use) :
  (∀ c : node, c ∈ cl -> g_constructor G c f from) -> a ∈ cl ->
  (f, use) ∈ U.get_fields_of_constructors F T is_local cl ->
  metadata_at sol (pattern_metadata ctx (get_single_field_source ctx a f) use)
    from.
Proof.
  intros Hcl Ha Hin. pose proof (least_closed_tables is_local F T Hleast) as Hc.
  pose proof (get_single_field_source_spec ctx a f) as Hs.
  cbn [plain_context rc_tables rc_facts] in Hs.
  destruct (gfoc_spec cl f use Hin) as [_ Hu].
  unfold pattern_metadata.
  destruct (get_single_field_source ctx a f) as [ | s | ].
  - intros b Hb Hhb. exfalso. destruct (Hs from (Hcl a Ha)) as [Hnas Hno].
    destruct Hb as (_ & [Has | Hsrc] & _); [ congruence | ].
    exact (Hno b Hsrc Hhb).
  - intros b Hb Hhb. cbn [describes].
    destruct (Hs from (Hcl a Ha)) as [Hnas Hone].
    destruct Hb as (_ & [Has | Hsrc] & _); [ congruence | ].
    exact (Hone b Hsrc Hhb).
  - destruct use as [ | flow_to].
    + destruct Hu as (c & Hcin & Htop).
      pose proof (field_used_top_any_usage is_local F T Hleast sets c from f
                    Htop (Hcl c Hcin)) as Hafrom.
      intros b Hb _. destruct Hb as (_ & _ & _ & Hau & _).
      cbn [describes]. exact (Hau Hafrom).
    + destruct Hu as (_ & Hnot & Hys).
      intros b Hb Hhb. cbn [describes].
      destruct (s_any_usage sol b) eqn:Hab; [ left; reflexivity | right ].
      intros z Hz. apply get_direct_usages_iff in Hz as (y & Hy & Hyz & Hhz).
      destruct (proj1 (Hys y) Hy) as (c & Hcin & Hyc).
      destruct (field_used_as_alias is_local F T Hleast sets c from y f Hyc
                  (Hcl c Hcin) (Hnot c Hcin)) as (Hhy & Hnay & Halias).
      destruct Hb as (_ & _ & _ & Hau_n & Hus_n).
      assert (Hnaf : s_any_usage sol from = false).
      { apply not_true_iff_false. intros H. rewrite (Hau_n H) in Hab.
        discriminate Hab. }
      pose proof (usages_alias _ _ _ Hc y from z Hyz Hnay Halias Hnaf)
        as Hfz.
      pose proof (has_usage_alias _ _ _ Hc y from Hhy Halias) as Hhf.
      destruct (Hus_n Hhf z Hfz) as [Haz | Haz];
        [ split; assumption | exfalso; exact (bool_both _ Haz Hab) ].
Qed.

(* The level of a value slot unboxed in turn. *)
Lemma gfoc_level (dominated : list (node * node)) (to_unbox : list node)
    (e : expr) (cl : list node) (a from s : node) (f : field)
    (flow_to : list node) :
  (∀ c : node, c ∈ cl -> g_constructor G c f from) -> a ∈ cl ->
  s_has_usage sol from = true ->
  get_single_field_source ctx a f = Field_one_source s ->
  (f, U.Use_known flow_to) ∈ U.get_fields_of_constructors F T is_local cl ->
  forallb (U.has_to_be_unboxed dominated to_unbox) flow_to = true ->
  level_ok F T dominated to_unbox e from s (get_direct_usages ctx flow_to).
Proof.
  intros Hcl Ha Hhf Hsrc Hin Hall.
  pose proof (least_closed_tables is_local F T Hleast) as Hc.
  pose proof (get_single_field_source_spec ctx a f) as Hs.
  rewrite Hsrc in Hs. cbn [plain_context rc_tables rc_facts] in Hs.
  destruct (Hs from (Hcl a Ha)) as [Hnas Hone].
  destruct (gfoc_spec cl f _ Hin) as (_ & _ & Hnot & Hys).
  split; [ | split; [ exact Hhf | split; [ | split ] ] ].
  - intros b Hb Hhb. destruct Hb as (_ & [Has | Hsrc'] & _);
      [ congruence | exact (Hone b Hsrc' Hhb) ].
  - destruct (s_any_usage sol from) eqn:Haf; [ left; reflexivity | right ].
    intros z Hz. apply get_direct_usages_iff in Hz as (y & Hy & Hyz & _).
    destruct (proj1 (Hys y) Hy) as (c & Hcin & Hyc).
    destruct (field_used_as_alias is_local F T Hleast sets c from y f Hyc
                (Hcl c Hcin) (Hnot c Hcin)) as (_ & Hnay & Halias).
    exact (usages_alias _ _ _ Hc y from z Hyz Hnay Halias Haf).
  - intros z Hz. apply get_direct_usages_iff in Hz as (_ & _ & _ & Hhz).
    exact Hhz.
  - intros z Hz. apply get_direct_usages_iff in Hz as (y & Hy & Hyz & _).
    rewrite forallb_forall in Hall.
    destruct (unboxed_has F T dominated to_unbox e y (Hall y Hy))
      as (fs & Hfs).
    exists y, fs. split; [ exact Hfs | exact Hyz ].
Qed.

End Constructor_fields.

(* What the leaves of the closure types ask of a representation, of the
   closures [cl] of its set: some field of it is read from them, if it has
   one, so that the classification of the fields read is not empty; and a
   field unboxed in turn, read into [flow_to], is read into names that
   the analysis unboxes, with the fields that it reads from their uses
   ([mk_shape]), under the premises of the unboxed result types. *)
Definition repr_leaves_ok (is_local : field -> bool) (F : facts) (T : tables)
    (cm : finite_map variable) (cl : list node) (fields : U.unboxed_fields)
    : Prop :=
  (fields <> [] ->
   ∃ f : field,
     f ∈ map fst fields
     ∧ Keyed.is_bound field_eqb f
         (U.get_fields_of_constructors F T is_local cl)
       = true)
  ∧ ∀ (f : field) (nested : U.unboxed_fields) (flow_to : list node),
      (f, U.Unboxed nested) ∈ fields ->
      (f, U.Use_known flow_to)
        ∈ U.get_fields_of_constructors F T is_local cl ->
      ∃ (dominated : list (node * node)) (to_unbox : list node) (e : expr),
        cm = U.bound_code_my_closures e
        ∧ block_fields_of_one_kind (graph_of_facts F)
        ∧ UCS.check_unboxed_uses F T
            (fst (U.unit_unboxed_fields F T dominated to_unbox e))
          = true
        ∧ forallb (U.has_to_be_unboxed dominated to_unbox) flow_to = true
        ∧ mk_shape F T dominated to_unbox e
            (get_direct_usages (plain_context is_local F T []) flow_to)
            nested.

(* The same of the representation of each closure of a set. *)
Definition reprs_leaves_ok (is_local : field -> bool) (F : facts)
    (T : tables) (cm : finite_map variable)
    (reprs : list (node * U.closure_repr)) : Prop :=
  ∀ (fdecls : list (function_slot * code_id)) (sites : function_slot -> node)
    (slot : function_slot) (id : code_id) (r : U.closure_repr)
    (cl : list node),
    lookup slot fdecls = Some id -> set_sites (graph_of_facts F) fdecls sites ->
    Keyed.lookup node_eqb (sites slot) reprs = Some r ->
    (∀ b : node,
       b ∈ cl <-> ∃ g : function_slot, is_bound g fdecls = true ∧ b = sites g)
    -> repr_leaves_ok is_local F T cm cl (U.cr_fields r).

(* A representation with no field asks nothing. *)
Lemma reprs_leaves_ok_of_no_slots (is_local : field -> bool) (F : facts)
    (T : tables) (cm : finite_map variable)
    (reprs : list (node * U.closure_repr)) :
  reprs_without_slots reprs -> reprs_leaves_ok is_local F T cm reprs.
Proof.
  intros Hno fdecls sites slot id r cl _ _ Hlr _.
  rewrite (Hno (sites slot) r Hlr).
  split; [ intros H; contradiction | intros f nested flow_to [] ].
Qed.

(* The case of a field in [changed_leaves], for a classification [fg]. *)
Definition changed_field_leaves (ctx : rewrite_context)
    (cm : finite_map variable) (source : node)
    (reads : list (field * U.field_use)) (fg : bool)
    (p : field * U.unboxed_field) : list unboxed_leaf :=
  let '(f, u) := p in
  match Keyed.lookup field_eqb f reads with
  | None => forgotten_leaves f u
  | Some use =>
    if fg then forgotten_leaves f u
    else
      let source_f := get_single_field_source ctx source f in
      match u with
      | U.Not_unboxed _ =>
        [([f], U.field_kind f, Some (pattern_metadata ctx source_f use))]
      | U.Unboxed _ =>
        match use, source_f with
        | U.Use_known flow_to, Field_one_source s =>
          map (fun l => let '(p, k, m) := l in (f :: p, k, m))
            (unboxed_leaves ctx cm (U.field_kind f)
               (get_direct_usages ctx flow_to) s u)
        | _, _ => forgotten_leaves f u
        end
      end
  end.

Lemma changed_leaves_eq (ctx : rewrite_context) (cm : finite_map variable)
    (source : node) (reads : list (field * U.field_use))
    (fields : U.unboxed_fields) :
  changed_leaves ctx cm source reads fields
  = match
      classify_field_map
        (filter (fun f => Keyed.is_bound field_eqb f reads) (map fst fields))
    with
    | Class_empty => []
    | cls =>
      flat_map (changed_field_leaves ctx cm source reads (forgets_class cls))
        fields
    end.
Proof. unfold changed_leaves. destruct (classify_field_map _); reflexivity. Qed.

(* Its leaves are those of the fields, in order. *)
Lemma changed_field_leaves_paths (ctx : rewrite_context)
    (cm : finite_map variable) (source : node)
    (reads : list (field * U.field_use)) (fg : bool) :
  ∀ fields : U.unboxed_fields,
    map (fun l => let '(p, k, _) := l in (p, k))
      (flat_map (changed_field_leaves ctx cm source reads fg) fields)
    = unboxed_field_paths Value (U.Unboxed fields).
Proof.
  induction fields as [ | [f u] fields IH]; [ reflexivity | ].
  change (unboxed_field_paths Value (U.Unboxed ((f, u) :: fields)))
    with (map (fun pk => (f :: fst pk, snd pk))
            (unboxed_field_paths (U.field_kind f) u)
          ++ unboxed_field_paths Value (U.Unboxed fields)).
  change (flat_map (changed_field_leaves ctx cm source reads fg)
            ((f, u) :: fields))
    with (changed_field_leaves ctx cm source reads fg (f, u)
          ++ flat_map (changed_field_leaves ctx cm source reads fg) fields).
  rewrite map_app. f_equal; [ | exact IH ].
  assert (Hforgot : map (fun l => let '(p, k, _) := l in (p, k))
                      (forgotten_leaves f u)
                    = map (fun pk => (f :: fst pk, snd pk))
                        (unboxed_field_paths (U.field_kind f) u)).
  { unfold forgotten_leaves. rewrite map_map. reflexivity. }
  unfold changed_field_leaves.
  destruct (Keyed.lookup field_eqb f reads) as [use | ]; [ | exact Hforgot ].
  destruct fg; [ exact Hforgot | ]. cbv zeta.
  destruct u as [v | nested]; [ reflexivity | ].
  destruct use as [ | flow_to]; [ exact Hforgot | ].
  destruct (get_single_field_source ctx source f) as [ | s | ];
    try exact Hforgot.
  rewrite map_map.
  rewrite <- (unboxed_leaves_paths ctx cm (U.Unboxed nested) (U.field_kind f)
                (get_direct_usages ctx flow_to) s).
  rewrite map_map. apply map_ext. intros [[p k] m]. reflexivity.
Qed.

(* ======================================================================== *)
(* Closures whose representation changes, for every closure                 *)
(* ======================================================================== *)

Section Changed_sound.

Variable fx : fixes.
Hypothesis Hfix_boxed : fix_boxed fx = true.
Hypothesis Hfix_fields : fix_alias fx = true ∨ fix_deleted_field fx = true.
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
Variable loose : bool.
Variable cm : finite_map variable.
Variable reprs : list (node * U.closure_repr).

Local Abbreviation ctx := (plain_context is_local F T sets).
Local Abbreviation ctxu := (Rewrite_context F T is_local sets (map fst reprs)).
Local Abbreviation G := (graph_of_facts F).
Local Abbreviation sol := (solution_of_tables T).

Hypothesis Hc : closed is_local G sol.
Hypothesis Hleast : least_either is_local F T.
Hypothesis Hlocal : is_local_slots_only is_local.
Hypothesis Hsets : sets_of_graph G sets.
Hypothesis Henv : typing_env_satisfied_strictly statics valuation tenv.
Hypothesis Hfrag : env_in_fragment_with loose valuation tenv.
Hypothesis Halloc : statics_allocated sol statics.
Hypothesis Hst : statics_rebuilt G sol statics statics'.
Hypothesis Hcopies : copies_valued tenv canon valuation valuation'.
Hypothesis Hwhole : loose = true -> static_sets_whole sol statics statics'.
Hypothesis Hunst : changed_unstored G sol reprs.
Hypothesis Hreprs : reprs_of_sets G reprs.
Hypothesis Hleaves : reprs_leaves_ok is_local F T cm reprs.

(* No allocation that may come to [n] changes. *)
Definition unchanged_at (n : node) : Prop :=
  ∀ a : node,
    may_come_from sol a n -> s_has_source sol a = true ->
    Keyed.lookup node_eqb a reprs = None.

(* [closure_under], with the allocations [sites] of the set and the slot
   [slot] of the closure named. *)
Definition closure_with (sites : function_slot -> node) (slot : function_slot)
    (m : rewrite_metadata) (v v' : value) : Prop :=
  ∃ (fdecls : list (function_slot * code_id))
    (vs vs' : list (value_slot * value)),
    set_closure_pair statics statics' slot fdecls vs vs' v v'
    ∧ is_bound slot fdecls = true ∧ set_sites G fdecls sites
    ∧ for_all_bindings
        (value_slot_rebuilt G sol (rebuilt_value G sol) fdecls sites vs') vs
    ∧ describes sol m (sites slot).

(* A value of the first run and what the third has for it, which the
   second has too: as [rebuilt_ext] relates them, where no allocation
   changes. *)
Definition unchanged_ext (m : rewrite_metadata) (v v' : value) : Prop :=
  (∃ c : const, v = Value_const c ∧ v' = v)
  ∨ (∃ n : node,
       rebuilt_value G sol v n v' ∧ metadata_at sol m n
       ∧ (forgets m = false -> s_has_usage sol n = true) ∧ unchanged_at n)
  ∨ (∃ (sites : function_slot -> node) (slot : function_slot),
       closure_with sites slot m v v'
       ∧ Keyed.lookup node_eqb (sites slot) reprs = None).

(* A closure of a set that changes, rebuilt with its set ([vs_mid] in the
   second run), and the closure of the third run in the representation
   of its allocation. *)
Definition changed_ext (m : rewrite_metadata) (v v' : value) : Prop :=
  ∃ (slot : function_slot) (fdecls : list (function_slot * code_id))
    (vs vs_mid : list (value_slot * value)) (sites : function_slot -> node)
    (r : U.closure_repr),
    v = Value_closure_of_set slot fdecls vs
    ∧ is_bound slot fdecls = true ∧ set_sites G fdecls sites
    ∧ for_all_bindings
        (value_slot_rebuilt G sol (rebuilt_value G sol) fdecls sites vs_mid)
        vs
    ∧ describes sol m (sites slot)
    ∧ Keyed.lookup node_eqb (sites slot) reprs = Some r
    ∧ changed_closure_rebuilt r (Value_closure_of_set slot fdecls vs_mid) v'.

Definition rel_u (m : rewrite_metadata) (v v' : value) : Prop :=
  unchanged_ext m v v' ∨ changed_ext m v v'.

Lemma unchanged_rebuilt (m : rewrite_metadata) (v v' : value) :
  unchanged_ext m v v' -> rebuilt_ext F T statics statics' m v v'.
Proof.
  intros [H | [(n & H1 & H2 & H3 & _)
              | (sites & slot & (fdecls & vs & vs' & H1 & H2 & H3 & H4 & H5)
                 & _)]].
  - left. left. exact H.
  - left. right. exists n. auto.
  - right. exists slot, fdecls, vs, vs', sites. auto.
Qed.

Lemma changed_closures (m : rewrite_metadata) (v v' : value) :
  changed_ext m v v' ->
  (∃ slot fdecls vs, v = Value_closure_of_set slot fdecls vs)
  ∧ ∃ slot' fdecls' vs', v' = Value_closure_of_set slot' fdecls' vs'.
Proof.
  intros (slot & fdecls & vs & vs_mid & sites & r & -> & _ & _ & _ & _ & _
          & Hcr).
  split; [ eauto | ].
  destruct Hcr as (d & a & b & _ & -> & _). eauto.
Qed.

Lemma rel_u_kind (m : rewrite_metadata) (v v' : value) :
  rel_u m v v' -> kind_of_value v' = kind_of_value v.
Proof.
  intros [H | H].
  - exact (ext_kind F T statics statics' m v v' (unchanged_rebuilt m v v' H)).
  - destruct (changed_closures m v v' H)
      as [(s & d & vs & ->) (s' & d' & vs' & ->)].
    reflexivity.
Qed.

Lemma rel_u_const (m : rewrite_metadata) (c : const) (v' : value) :
  rel_u m (Value_const c) v' -> v' = Value_const c.
Proof.
  intros [H | H].
  - exact (ext_const F T statics statics' m c v'
             (unchanged_rebuilt _ _ _ H)).
  - destruct (changed_closures _ _ _ H) as [(s & d & vs & Habs) _].
    discriminate Habs.
Qed.

Lemma rel_u_of_const (m : rewrite_metadata) (c : const) :
  rel_u m (Value_const c) (Value_const c).
Proof. left. left. exists c. split; reflexivity. Qed.

Lemma rel_u_unallocated (m : rewrite_metadata) (v v' : value) :
  unallocated v -> rel_u m v v' -> v' = v.
Proof.
  intros Hu [H | H].
  - exact (ext_unallocated F T statics statics' m v v' Hu
             (unchanged_rebuilt _ _ _ H)).
  - destruct (changed_closures _ _ _ H) as [(s & d & vs & ->) _].
    destruct Hu as [(c & Hc') | (s0 & Hs)]; discriminate.
Qed.

Lemma rel_u_not_null (m : rewrite_metadata) (v v' : value) :
  rel_u m v v' -> v <> Value_const Null -> v' <> Value_const Null.
Proof.
  intros [H | H].
  - exact (ext_not_null F T statics statics' m v v'
             (unchanged_rebuilt _ _ _ H)).
  - destruct (changed_closures _ _ _ H) as [_ (s & d & vs & ->)].
    intros _. discriminate.
Qed.

Lemma block_rebuilt_u (m : rewrite_metadata) (v v' : value) (tag : Z)
    (fields : list value) :
  forgets m = false -> rel_u m v v' ->
  block_of statics v = Some (tag, fields) ->
  ∃ (a : node) (fields' : list value),
    describes sol m a ∧ block_of statics' v' = Some (tag, fields')
    ∧ fields_rebuilt (field_rebuilt G sol (rebuilt_value G sol) a) 0 fields
        fields'.
Proof.
  intros Hm [H | H] Hb.
  - exact (block_rebuilt F T statics statics' Halloc Hst m v v' tag fields Hm
             (unchanged_rebuilt _ _ _ H) Hb).
  - destruct (changed_closures _ _ _ H) as [(s & d & vs & ->) _].
    discriminate Hb.
Qed.

Lemma boxed_rebuilt_u (m : rewrite_metadata) (v v' : value)
    (k : boxable_number) (c : const) :
  forgets m = false -> rel_u m v v' ->
  boxed_number_of statics v = Some c -> kind_of_const c = unboxed_kind k ->
  forgets (follow_field ctx m (Field_boxed_number k)) = false ->
  boxed_number_of statics' v' = Some c.
Proof.
  intros Hm [H | H] Hb Hk Hff.
  - exact (boxed_rebuilt is_local F T sets statics statics' Hc Hlocal Hst m v
             v' k c Hm (unchanged_rebuilt _ _ _ H) Hb Hk Hff).
  - destruct (changed_closures _ _ _ H) as [(s & d & vs & ->) _].
    discriminate Hb.
Qed.

Lemma string_rebuilt_u (m : rewrite_metadata) (v v' : value)
    (bytes : list Z) :
  forgets m = false -> rel_u m v v' ->
  string_of statics v = Some bytes -> string_of statics' v' = Some bytes.
Proof.
  intros Hm [H | H] Hs.
  - exact (string_rebuilt F T statics statics' Hst m v v' bytes Hm
             (unchanged_rebuilt _ _ _ H) Hs).
  - destruct (changed_closures _ _ _ H) as [(s & d & vs & ->) _].
    discriminate Hs.
Qed.

(* What is held in a field other than a function slot does not change. *)
Lemma unchanged_field (a from : node) (f : field) :
  g_constructor G a f from -> (∀ g : function_slot, f <> Field_function_slot g)
  -> unchanged_at from.
Proof.
  intros Hcons Hf c Hmc Hhc. exact (Hunst a f from c Hcons Hf Hmc Hhc).
Qed.

Definition rw_sound_u
    (rw : rewrite_metadata -> flambda_type -> flambda_type) : Prop :=
  ∀ (m : rewrite_metadata) (t : flambda_type) (v v' : value),
    type_in_fragment_with loose valuation t -> rel_u m v v' ->
    type_satisfied statics valuation v t ->
    type_satisfied statics' valuation' v' (rw m t).

Section With_rw_u.

Variable rw : rewrite_metadata -> flambda_type -> flambda_type.
Hypothesis Hrw : rw_sound_u rw.
Hypothesis Hrwf :
  fix_deleted_field fx = true
  ∨ rw_forgets statics statics' valuation valuation' loose rw.

Lemma field_sound_u (m : rewrite_metadata) (a : node) (k : nat)
    (field field' : value) (ty : flambda_type) :
  forgets m = false -> describes sol m a ->
  field_rebuilt G sol (rebuilt_value G sol) a k field field' ->
  type_in_fragment_with loose valuation ty ->
  type_satisfied statics valuation field ty ->
  type_satisfied statics' valuation' field' (rewrite_field fx ctx rw m k ty).
Proof.
  intros Hm Hd (from & Hcons & Hused & Hunused) Hfr Hsat.
  assert (Hk : kind_of_value field' = kind_of_value field).
  { destruct (s_field_used sol a (Field_block k (kind_of_value field)));
      [ exact (rebuilt_value_kind G sol _ _ _ (proj2 (Hused eq_refl)))
      | rewrite (Hunused eq_refl); reflexivity ]. }
  unfold rewrite_field, block_slot. cbv zeta.
  destruct (deleted_field fx (follow_field ctx m (Field_block k (kind ty))))
    eqn:Hdel; [ exact (sat_unknown_of_kind _ _ _ _ field field' ty Hsat Hk) | ].
  rewrite <- (type_satisfied_kind _ _ _ _ Hsat) in Hdel |- *.
  destruct (s_field_used sol a (Field_block k (kind_of_value field)))
    eqn:Hfu.
  - destruct (Hused eq_refl) as [Hhu Hrv].
    apply (Hrw _ ty field field' Hfr); [ | exact Hsat ].
    left. right. left. exists from.
    split; [ exact Hrv | split; [ | split; [ intros _; exact Hhu | ] ] ].
    + exact (follow_field_at ctx is_local Hc m a from _ Hd Hm Hcons
               (not_local_block is_local Hlocal k _) Hfu).
    + apply (unchanged_field a from _ Hcons). intros g Habs.
      discriminate Habs.
  - rewrite (Hunused eq_refl).
    pose proof (follow_field_unused ctx is_local Hc m a from _ Hd Hm Hcons
                  (not_local_block is_local Hlocal k _) Hfu) as Hff.
    rewrite Hff in Hdel |- *.
    destruct Hrwf as [Hdf | Hrwf'];
      [ unfold deleted_field in Hdel; rewrite Hdf in Hdel;
        discriminate Hdel | ].
    apply (Hrwf' At_least_one_source_no_usages ty field _ eq_refl Hfr Hsat).
    reflexivity.
Qed.

Lemma fields_sound_u (m : rewrite_metadata) (a : node) :
  forgets m = false -> describes sol m a ->
  ∀ (tys : list flambda_type) (k : nat) (fields fields' : list value),
    fields_rebuilt (field_rebuilt G sol (rebuilt_value G sol) a) k fields
      fields' ->
    (∀ ty : flambda_type,
       ty ∈ tys -> type_in_fragment_with loose valuation ty) ->
    fields_satisfied (type_satisfied statics valuation) tys fields ->
    fields_satisfied (type_satisfied statics' valuation')
      (map (fun p => rewrite_field fx ctx rw m (fst p) (snd p))
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
    + exact (field_sound_u m a k field field' ty Hm Hd Hfield
               (Hfr ty (or_introl eq_refl)) Hf).
    + exact (IH (S k) fields fields' Hfields
               (fun ty' H => Hfr ty' (or_intror H)) Hfs).
Qed.

Lemma block_case_sound_u (m : rewrite_metadata) (a : node) (tag : Z)
    (fields fields' : list value) (case : row_like_block_case) :
  forgets m = false -> describes sol m a ->
  fields_rebuilt (field_rebuilt G sol (rebuilt_value G sol) a) 0 fields
    fields' ->
  (∀ ty : flambda_type,
     ty ∈ case_maps_to case -> type_in_fragment_with loose valuation ty) ->
  row_like_block_case_satisfied statics valuation tag fields case ->
  row_like_block_case_satisfied statics' valuation' tag fields'
    (rewrite_block_case fx ctx rw m case).
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
    exact (fields_sound_u m a Hm Hd tys 0 fields fields' Hfields Hfr Hfs).
Qed.

Lemma blocks_sound_u (m : rewrite_metadata) (a : node) (tag : Z)
    (fields fields' : list value) (blocks : row_like_for_blocks) :
  forgets m = false -> describes sol m a ->
  fields_rebuilt (field_rebuilt G sol (rebuilt_value G sol) a) 0 fields
    fields' ->
  shaped_blocks_with loose blocks = true
  ∧ (∀ (s : simple) (k : flambda_kind) (w : value),
       (s, k) ∈ aliases_of_row_like_for_blocks blocks ->
       eval_simple valuation s = Some w -> unallocated w) ->
  row_like_for_blocks_satisfied statics valuation tag fields blocks ->
  row_like_for_blocks_satisfied statics' valuation' tag fields'
    (rewrite_row_like_for_blocks fx ctx rw m blocks).
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
  - apply (block_case_sound_u m a tag fields fields' case Hm Hd Hfields);
      [ | exact Hsat ].
    apply (known_case_fragment valuation known other am tag case Hcf Hal).
    exact (keyed_lookup_in _ Z.eqb_eq tag _ known Hl).
  - exact I.
  - destruct other as [case | ]; [ | contradiction ].
    exact (block_case_sound_u m a tag fields fields' case Hm Hd Hfields
             (other_case_fragment valuation known case am Hcf Hal) Hsat).
Qed.

Lemma non_null_sound_u (m : rewrite_metadata)
    (head : head_of_kind_value_non_null) (v v' : value) :
  forgets m = false -> non_null_in_fragment loose valuation head ->
  rel_u m v v' -> v <> Value_const Null ->
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
    + left. rewrite (rel_u_const m _ v' Hv). exists i.
      split; [ reflexivity | split; [ exact I | split; [ | exact I ] ] ].
      destruct immediates as [ty | ]; [ | exact I ].
      apply (Hrw m ty (Value_const (Naked_immediate i))
               (Value_const (Naked_immediate i))).
      * exact (immediates_fragment valuation is_int ty get_tag blocks
                 extensions is_unique Hfr).
      * apply rel_u_of_const.
      * exact Himm.
    + destruct (block_rebuilt_u m v v' tag fields Hm Hv Hblock)
        as (a & fields' & Hd & Hblock' & Hfields).
      right. exists tag, fields'.
      split; [ exact Hblock' | split; [ exact I | split; [ exact I | ] ] ].
      split; [ | exact I ].
      destruct blocks as [blocks | ]; [ | exact I ].
      exact (blocks_sound_u m a tag fields fields' blocks Hm Hd Hfields
               (blocks_fragment valuation is_int immediates get_tag blocks
                  extensions is_unique Hfr)
               Hblocks).
  - rewrite mutable_block_head_satisfied in Hsat. contradiction.
  - cbn [rewrite_head_of_kind_value_non_null]. rewrite Hfix_boxed.
    rewrite boxed_number_satisfied in Hsat.
    destruct Hsat as (c & Hbox & Hkc & Hcontents).
    destruct (forgets (follow_field ctx m (Field_boxed_number k))) eqn:Hff;
      cbn [andb].
    + rewrite (rel_u_kind m v v' Hv).
      exact (boxed_number_of_kind statics v c Hbox).
    + rewrite boxed_number_satisfied. exists c.
      split; [ exact (boxed_rebuilt_u m v v' k c Hm Hv Hbox Hkc Hff) | ].
      split; [ exact Hkc | ].
      apply (Hrw m contents (Value_const c) (Value_const c)
               (contents_fragment valuation k contents am Hfr));
        [ apply rel_u_of_const | exact Hcontents ].
  - cbn [rewrite_head_of_kind_value_non_null].
    destruct closures as [[ | b known]].
    + rewrite closures_satisfied in Hsat.
      destruct Hsat as (slot & fdecls & vs & _ & Hrow).
      rewrite row_like_for_closures_lookup_iff in Hrow.
      destruct Hrow as (case & Hl & _). discriminate Hl.
    + rewrite (rel_u_kind m v v' Hv).
      exact (proj1 (head_non_null_kind _ _ _ _ Hsat)).
  - cbn [rewrite_head_of_kind_value_non_null].
    rewrite string_satisfied in Hsat |- *.
    destruct Hsat as (bytes & Hstr & Hin). exists bytes.
    split; [ exact (string_rebuilt_u m v v' bytes Hm Hv Hstr) | exact Hin ].
Qed.

Lemma head_value_sound_u (m : rewrite_metadata) (head : head_of_kind_value)
    (v v' : value) :
  forgets m = false ->
  type_in_fragment_with loose valuation (Type_value (Ok (No_alias head))) ->
  rel_u m v v' ->
  head_of_kind_value_satisfied statics valuation v head ->
  head_of_kind_value_satisfied statics' valuation' v'
    (rewrite_head_of_kind_value fx ctx rw m head).
Proof.
  intros Hm Hfr Hv Hsat. destruct head as [non_null is_null].
  cbn [rewrite_head_of_kind_value].
  rewrite head_of_kind_value_satisfied_eq in Hsat |- *.
  destruct Hsat as [[Hnull His] | [Hnn [Hpart His]]].
  - left. subst v. rewrite (rel_u_const m Null v' Hv).
    split; [ reflexivity | ].
    destruct is_null; [ contradiction | exact I ].
  - right. split; [ exact (rel_u_not_null m v v' Hv Hnn) | ].
    split; [ | destruct is_null; exact I ].
    destruct non_null as [ | h | ]; [ | | contradiction ].
    + rewrite (rel_u_kind m v v' Hv). exact Hpart.
    + exact (non_null_sound_u m h v v' Hm
               (head_fragment valuation h is_null Hfr) Hv Hnn Hpart).
Qed.

Lemma expanded_sound_u (m : rewrite_metadata) (e : Expanded_type.t)
    (v v' : value) :
  forgets m = false ->
  type_in_fragment_with loose valuation (Expanded_type.to_type e) ->
  rel_u m v v' ->
  type_satisfied statics valuation v (Expanded_type.to_type e) ->
  type_satisfied statics' valuation' v' (rewrite_expanded_head fx ctx rw m e).
Proof.
  intros Hm Hfr Hv Hsat.
  destruct e as [k [ | h | ]]; unfold rewrite_expanded_head;
    cbn [Expanded_type.descr Expanded_type.kind Expanded_type.to_type] in *.
  - apply unknown_satisfied. rewrite (rel_u_kind m v v' Hv).
    exact (proj1 (unknown_satisfied _ _ _ _) Hsat).
  - destruct h as [hv | hi | kn ns | hr | hreg].
    + rewrite type_satisfied_value in Hsat |- *.
      exact (head_value_sound_u m hv v v' Hm Hfr Hv Hsat).
    + rewrite type_satisfied_naked_immediate in Hsat |- *.
      cbn [descr_satisfied] in Hsat |- *.
      destruct Hsat as (i & -> & Hnum & _).
      rewrite (rel_u_const m _ v' Hv). exists i.
      split; [ reflexivity | split; [ exact Hnum | ] ].
      intros relation n [].
    + rewrite type_satisfied_naked_number in Hsat |- *.
      cbn [descr_satisfied] in Hsat |- *.
      destruct Hsat as (c & n & -> & Hk & Hn & Hin).
      rewrite (rel_u_const m _ v' Hv). exists c, n. auto.
    + rewrite type_satisfied_rec_info in Hsat |- *.
      cbn [descr_satisfied] in Hsat |- *.
      rewrite (rel_u_kind m v v' Hv). exact Hsat.
    + rewrite type_satisfied_region in Hsat |- *.
      cbn [descr_satisfied] in Hsat |- *.
      rewrite (rel_u_kind m v v' Hv). exact Hsat.
  - exfalso. exact (bottom_satisfied _ _ _ _ Hsat).
Qed.

Lemma identity_sound_u (m : rewrite_metadata) (t : flambda_type)
    (v v' : value) :
  forgets m = false -> type_in_fragment_with loose valuation t ->
  rel_u m v v' -> type_satisfied statics valuation v t ->
  type_satisfied statics' valuation' v' (rewrite_identity fx ctx tenv rw m t).
Proof.
  intros Hm Ht Hv Hsat. unfold rewrite_identity.
  destruct (expand_head tenv t) as [e | ] eqn:He;
    [ | exact (sat_unknown_of_kind statics statics' valuation valuation' v v'
                 t Hsat (rel_u_kind m v v' Hv)) ].
  pose proof (expand_head_sound statics valuation tenv t e v Henv Hsat He)
    as Hsat_e.
  pose proof (expand_head_fragment valuation tenv Hfrag t e He Ht) as Hfr_e.
  pose proof (expanded_sound_u m e v v' Hm Hfr_e Hv Hsat_e) as Hmain.
  destruct (must_be_singleton (rewrite_expanded_head fx ctx rw m e))
    as [[c | ] | ] eqn:Hmb; try exact Hmain.
  pose proof (must_be_singleton_satisfied _ _ v' _ c Hmb Hmain) as ->.
  apply alias_satisfied.
  split; [ reflexivity | exact (type_satisfied_kind _ _ _ _ Hmain) ].
Qed.

End With_rw_u.

(* The rewriting with changes, under a metadata that forgets. *)
Lemma changed_rule_forgets (m : rewrite_metadata) (t : flambda_type) :
  forgets m = true -> changed_rule fx ctxu tenv reprs m t = None.
Proof. intros H. unfold changed_rule. rewrite H. reflexivity. Qed.

Lemma rewrite_forgets_u (Hfix_alias : fix_alias fx = true) (fuel : nat) :
  rw_forgets statics statics' valuation valuation' loose
    (rewrite_arbitrary_u fx ctxu tenv canon cm reprs fuel)
  ∧ rw_forgets statics statics' valuation valuation' loose
      (rewrite_u fx ctxu tenv canon cm reprs fuel).
Proof.
  induction fuel as [ | fuel [IHa IHr]]; split; intros m t v v' Hm Ht Hv Hk.
  - exact (sat_unknown_of_kind _ _ _ _ v v' t Hv Hk).
  - exact (sat_unknown_of_kind _ _ _ _ v v' t Hv Hk).
  - cbn [rewrite_arbitrary_u].
    destruct (get_alias_opt t) as [s | ] eqn:Ha;
      [ | exact (IHr m t v v' Hm Ht Hv Hk) ].
    pose proof (proj1 (get_alias_satisfied _ _ v t s Ha) Hv) as [Hs Hkt].
    destruct (Typing_env.get_canonical_simple_exn tenv
                (Some Name_mode.in_types) None s)
      as [[[x | sym | c] | ] | ] eqn:Hcan;
      try (rewrite Hfix_alias, Hm;
           exact (sat_unknown_of_kind _ _ _ _ v v' t Hv Hk));
      try exact (sat_unknown_of_kind _ _ _ _ v v' t Hv Hk).
    pose proof (canonical_eval tenv statics valuation Henv s (Const c) Hcan)
      as Hceq.
    rewrite Hs in Hceq. cbn in Hceq. injection Hceq as <-.
    apply (IHr m (type_for_const c) (Value_const c) v' Hm).
    + apply type_for_const_fragment.
    + apply type_for_const_satisfied.
    + exact Hk.
  - cbn [rewrite_u]. rewrite (changed_rule_forgets m t Hm).
    unfold rewrite_rule. rewrite Hm.
    exact (sat_unknown_of_kind _ _ _ _ v v' t Hv Hk).
Qed.

(* At a field of a block that no use reads, as [fields_forget]. *)
Lemma fields_forget_u (fuel : nat) :
  fix_deleted_field fx = true
  ∨ rw_forgets statics statics' valuation valuation' loose
      (rewrite_arbitrary_u fx ctxu tenv canon cm reprs fuel).
Proof.
  destruct Hfix_fields as [Ha | Hd]; [ right | left; exact Hd ].
  exact (proj1 (rewrite_forgets_u Ha fuel)).
Qed.

(* [closure_under_view], with the allocations named. *)
Lemma closure_with_view (sites : function_slot -> node)
    (slot0 : function_slot) (m : rewrite_metadata) (v v' : value)
    (slot : function_slot) (fdecls : list (function_slot * code_id))
    (vs : list (value_slot * value)) :
  closure_with sites slot0 m v v' ->
  closure_in_set statics v = Some (slot, fdecls, vs) ->
  ∃ vs' : list (value_slot * value),
    slot0 = slot
    ∧ closure_in_set statics' v' = Some (slot, fdecls, vs')
    ∧ describes sol m (sites slot)
    ∧ set_sites G fdecls sites
    ∧ for_all_bindings
        (value_slot_rebuilt G sol (rebuilt_value G sol) fdecls sites vs') vs
    ∧ ∀ (g : function_slot) (c' : value),
        is_bound g fdecls = true ->
        project_function_slot statics' slot g v' = Some c' ->
        ∃ c : value,
          project_function_slot statics slot g v = Some c
          ∧ (unallocated c -> c' = c)
          ∧ ∀ mg : rewrite_metadata,
              describes sol mg (sites g) -> closure_with sites g mg c c'.
Proof.
  intros (fdecls0 & vs0 & vs' & Hp & Hb & Hsites & Hslots & Hd) Hcis.
  destruct (closure_pair_in_set statics statics' _ _ _ _ _ _ Hp Hb)
    as [Hcis0 Hcis'].
  rewrite Hcis0 in Hcis. injection Hcis as -> -> ->.
  exists vs'.
  split; [ reflexivity | split; [ exact Hcis' | ] ].
  split; [ exact Hd | split; [ exact Hsites | split; [ exact Hslots | ] ] ].
  destruct (proj1 (is_bound_lookup slot fdecls) Hb) as [id Hl].
  intros g c' Hg Hpc'.
  destruct (proj1 (is_bound_lookup g fdecls) Hg) as [idg Hlg].
  destruct Hp as [[-> ->] | (csyms & s & -> & -> & Hs & Hall)].
  - cbn [project_function_slot] in Hpc' |- *.
    rewrite Pos.eqb_refl, Hl, Hlg in Hpc' |- *. cbn in Hpc' |- *.
    injection Hpc' as <-.
    exists (Value_closure_of_set g fdecls vs).
    split; [ reflexivity | split ].
    + intros [(c0 & Hc0) | (s0 & Hs0)]; discriminate.
    + intros mg Hdg. exists fdecls, vs, vs'.
      split; [ left; split; reflexivity | ].
      split; [ exact Hg | split; [ exact Hsites | split; [ exact Hslots | ] ] ].
      exact Hdg.
  - destruct (Hall slot s Hs) as [H1 H2].
    cbn [project_function_slot] in Hpc' |- *.
    rewrite H1. rewrite H2 in Hpc'.
    rewrite Pos.eqb_refl, Hl, Hlg in Hpc' |- *. cbn in Hpc' |- *.
    destruct (lookup g csyms) as [s_g | ] eqn:Hsg; [ | discriminate Hpc' ].
    cbn in Hpc' |- *. injection Hpc' as <-.
    exists (Value_symbol s_g). split; [ reflexivity | split; [ auto | ] ].
    intros mg Hdg. exists fdecls, vs, vs'.
    split; [ right; exists csyms, s_g; auto | ].
    split; [ exact Hg | split; [ exact Hsites | split; [ exact Hslots | ] ] ].
    exact Hdg.
Qed.

(* [closure_rebuilt_view] of a closure that does not change. *)
Lemma closure_rebuilt_view_u (m : rewrite_metadata) (v v' : value)
    (slot : function_slot) (fdecls : list (function_slot * code_id))
    (vs : list (value_slot * value)) :
  forgets m = false -> unchanged_ext m v v' ->
  closure_in_set statics v = Some (slot, fdecls, vs) ->
  ∃ (sites : function_slot -> node) (vs' : list (value_slot * value)),
    closure_in_set statics' v' = Some (slot, fdecls, vs')
    ∧ describes sol m (sites slot)
    ∧ set_sites G fdecls sites
    ∧ for_all_bindings
        (value_slot_rebuilt G sol (rebuilt_value G sol) fdecls sites vs') vs
    ∧ Keyed.lookup node_eqb (sites slot) reprs = None
    ∧ ∀ (g : function_slot) (c' : value),
        is_bound g fdecls = true ->
        project_function_slot statics' slot g v' = Some c' ->
        ∃ c : value,
          project_function_slot statics slot g v = Some c
          ∧ (unallocated c -> c' = c)
          ∧ (loose = true ->
             ∀ mg : rewrite_metadata,
               describes sol mg (sites g) -> closure_with sites g mg c c').
Proof.
  intros Hm [(c & -> & ->) | [(n & Hv & Hat & Hhu & Hun)
                             | (sites & slot0 & Hcw & Hnone)]] Hcis.
  { discriminate Hcis. }
  2:{ destruct (closure_with_view sites slot0 m v v' slot fdecls vs Hcw Hcis)
        as (vs' & -> & Hcis' & Hd & Hsites & Hslots & Hproj).
      exists sites, vs'.
      split; [ exact Hcis' | split; [ exact Hd | split; [ exact Hsites | ] ] ].
      split; [ exact Hslots | split; [ exact Hnone | ] ].
      intros g c' Hg Hpc'.
      destruct (Hproj g c' Hg Hpc') as (c & Hpc & Hu & Hcu').
      exists c.
      split; [ exact Hpc | split; [ exact Hu | intros _; exact Hcu' ] ]. }
  specialize (Hhu Hm).
  destruct v as [c | code vs0 | slot0 fdecls0 vs0 | s | tag fields | elements
                | c | l | callee args arity];
    cbn [closure_in_set] in Hcis; try discriminate;
    cbn [rebuilt_value] in Hv; try contradiction.
  - destruct (lookup slot0 fdecls0) as [id0 | ] eqn:Hl0; cbn in Hcis;
      [ | discriminate ].
    injection Hcis as -> -> ->.
    destruct Hv as (sites & vs' & -> & Hmcf & Hsites & Hslots).
    assert (Hb : is_bound slot fdecls = true)
      by (unfold is_bound; rewrite Hl0; reflexivity).
    assert (Hhs : s_has_source sol (sites slot) = true).
    { apply (has_source_function_slot _ _ _ Hc _ _ (sites slot)
               (proj2 (Hsites slot id0 Hl0 slot (sites slot))
                  (conj Hb eq_refl))
               eq_refl). }
    assert (Hd : describes sol m (sites slot)) by exact (Hat _ Hmcf Hhs).
    destruct (closure_with_view sites slot m _ _ slot fdecls vs
                (ex_intro _ fdecls (ex_intro _ vs (ex_intro _ vs'
                   (conj (or_introl (conj eq_refl eq_refl))
                      (conj Hb (conj Hsites (conj Hslots Hd)))))))
                (ltac:(cbn [closure_in_set]; rewrite Hl0; reflexivity)))
      as (vs'' & _ & Hcis' & Hd' & Hsites' & Hslots' & Hproj).
    exists sites, vs''.
    split; [ exact Hcis' | split; [ exact Hd' | split; [ exact Hsites' | ] ] ].
    split; [ exact Hslots' | split; [ exact (Hun _ Hmcf Hhs) | ] ].
    intros g c' Hg Hpc'. destruct (Hproj g c' Hg Hpc') as (c & Hpc & Hu & Hcu).
    exists c. split; [ exact Hpc | split; [ exact Hu | intros _; exact Hcu ] ].
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
      exists sites, vs'. split.
      { cbn [closure_in_set]. rewrite Hls', Hl0. reflexivity. }
      split.
      { rewrite Hsite. apply (Hat (Node_symbol s) Hmcf).
        exact (proj1 (Halloc s _ Hls)). }
      split; [ exact Hsites | split; [ exact Hslots | ] ].
      split.
      { rewrite Hsite. exact (Hun _ Hmcf (proj1 (Halloc s _ Hls))). }
      intros g c' Hg Hpc'.
      assert (Hpeq : project_function_slot statics' slot g (Value_symbol s)
                     = project_function_slot statics slot g (Value_symbol s)).
      { cbn [project_function_slot]. rewrite Hls, Hls'. reflexivity. }
      rewrite Hpeq in Hpc'. exists c'.
      split; [ exact Hpc' | split; [ auto | ] ].
      intros Hlo mg Hdg.
      destruct (Hwhole Hlo s slot csyms fdecls vs Hls (Hhus Hhu))
        as [Hs (vs'' & Hall)].
      assert (Hvs : vs'' = vs').
      { destruct (Hall slot s Hs) as [_ H2]. rewrite Hls' in H2.
        injection H2 as ->. reflexivity. }
      subst vs''.
      cbn [project_function_slot] in Hpc'. rewrite Hls in Hpc'.
      rewrite Pos.eqb_refl, Hl0 in Hpc'.
      destruct (proj1 (is_bound_lookup g fdecls) Hg) as [idg Hlg].
      rewrite Hlg in Hpc'. cbn in Hpc'.
      destruct (lookup g csyms) as [s_g | ] eqn:Hsg; [ | discriminate Hpc' ].
      cbn in Hpc'. injection Hpc' as <-.
      exists fdecls, vs, vs'.
      split; [ right; exists csyms, s_g; auto | ].
      split; [ exact Hg | split; [ exact Hsites | split; [ exact Hslots | ] ] ].
      exact Hdg.
Qed.

(* A node that is not a key of [reprs]. *)
Lemma lookup_not_mem (c : node) :
  mem_node c (map fst reprs) = false -> Keyed.lookup node_eqb c reprs = None.
Proof.
  intros H. destruct (Keyed.lookup node_eqb c reprs) as [r | ] eqn:Hl;
    [ exfalso | reflexivity ].
  apply (keyed_lookup_in node_eqb node_eqb_decides) in Hl.
  apply not_true_iff_false in H. apply H. apply mem_node_In.
  apply in_map_iff. exists (c, r). auto.
Qed.

Lemma existsb_none_changed (l : list set_of_closures) :
  existsb
    (fun set : set_of_closures =>
       existsb (fun p => mem_node (fst (snd p)) (rc_changed ctx)) set) l
  = false.
Proof.
  induction l as [ | set l IH]; [ reflexivity | ].
  cbn [existsb]. rewrite IH, orb_false_r.
  induction set as [ | p set IHs]; [ reflexivity | exact IHs ].
Qed.

(* The rule of a closure with changes is that without, where no set
   kept changes. *)
Lemma closure_rule_u (m : rewrite_metadata) (slot : function_slot)
    (am : Alloc_mode.t) (fts : finite_map (Or_unknown.t function_type))
    (cts vsts : finite_map flambda_type) (mn : bool) (spec : closure_rewrite)
    :
  closure_rule fx ctxu m slot am (Closures_entry fts cts vsts) mn
  = Rule_closure spec ->
  closure_rule fx ctx m slot am (Closures_entry fts cts vsts) mn
  = Rule_closure spec
  ∧ ∀ (mbl : must_be_local) (set : set_of_closures) (g : function_slot)
      (c : node) (code : Or_unknown.t code_id),
      set ∈ fst (refine_sets_of_closures fx ctx m slot
                   (map (fun g => (g, entry_code_id
                                        (Closures_entry fts cts vsts) g))
                      (domain cts))
                   mbl) ->
      (g, (c, code)) ∈ set -> Keyed.lookup node_eqb c reprs = None.
Proof.
  intros H.
  assert (Hr : ∀ (cids : list (function_slot * Or_unknown.t code_id))
                 (mbl : must_be_local),
             refine_sets_of_closures fx ctxu m slot cids mbl
             = refine_sets_of_closures fx ctx m slot cids mbl)
    by reflexivity.
  assert (Hfst : ∀ (cids : list (function_slot * Or_unknown.t code_id))
                   (mbl mbl' : must_be_local),
             fst (refine_sets_of_closures fx ctx m slot cids mbl)
             = fst (refine_sets_of_closures fx ctx m slot cids mbl')).
  { intros cids mbl mbl'. unfold refine_sets_of_closures. cbv zeta.
    destruct m; try reflexivity.
    match goal with
    | |- context [if is_nil ?l then _ else _] => destruct (is_nil l)
    end; reflexivity. }
  unfold closure_rule in H |- *. cbv zeta in H |- *.
  rewrite Hr in H.
  match goal with
  | H : (let '(_, _) := refine_sets_of_closures _ _ _ _ ?cids ?mb in _) = _
    |- _ =>
    set (mbi := mb) in H |- *;
    destruct (refine_sets_of_closures fx ctx m slot cids mbi)
      as [ls mbl0] eqn:Href
  end.
  cbn [rc_changed] in H.
  match type of H with
  | (if ?b then _ else _) = _ => destruct b eqn:Hex; [ discriminate H | ]
  end.
  rewrite existsb_none_changed. split; [ exact H | ].
  intros mbl set g c code Hset Hin.
  rewrite (Hfst _ mbl mbi) in Hset.
  rewrite Href in Hset. cbn [fst] in Hset.
  apply lookup_not_mem.
  apply not_true_iff_false. intros Hmem.
  apply not_true_iff_false in Hex. apply Hex.
  apply existsb_exists. exists set. split; [ exact Hset | ].
  apply existsb_exists. exists (g, (c, code)).
  split; [ exact Hin | exact Hmem ].
Qed.

Lemma rewrite_rule_u (m : rewrite_metadata) (t : flambda_type)
    (spec : closure_rewrite) :
  rewrite_rule fx ctxu tenv m t = Rule_closure spec ->
  rewrite_rule fx ctx tenv m t = Rule_closure spec
  ∧ ∀ (slot : function_slot) (am : Alloc_mode.t)
      (fts : finite_map (Or_unknown.t function_type))
      (cts vsts : finite_map flambda_type) (ft : function_type)
      (mbl : must_be_local) (set : set_of_closures) (g : function_slot)
      (c : node) (code : Or_unknown.t code_id),
      Provers.meet_single_closures_entry tenv t
      = Some (Meet_shortcut.Known_result
                (slot, am, Closures_entry fts cts vsts, ft)) ->
      set ∈ fst (refine_sets_of_closures fx ctx m slot
                   (map (fun g => (g, entry_code_id
                                        (Closures_entry fts cts vsts) g))
                      (domain cts))
                   mbl) ->
      (g, (c, code)) ∈ set -> Keyed.lookup node_eqb c reprs = None.
Proof.
  unfold rewrite_rule. destruct (forgets m); [ discriminate | ].
  destruct (kind t); try discriminate.
  destruct (Provers.meet_single_closures_entry tenv t)
    as [[[[[slot am] [fts cts vsts]] ft] | | ] | ] eqn:Hmeet; try discriminate.
  intros H. destruct (closure_rule_u m slot am fts cts vsts _ spec H)
    as [H1 H2].
  split; [ exact H1 | ].
  intros slot' am' fts' cts' vsts' ft' mbl set g c code Hm' Hset Hin.
  injection Hm' as <- <- <- <- <- <-. exact (H2 mbl set g c code Hset Hin).
Qed.

(* Aliases. *)
Lemma rewrite_arbitrary_u_alias (fuel : nat) :
  rw_sound_u (rewrite_arbitrary_u fx ctxu tenv canon cm reprs fuel) ->
  rw_alias statics statics' valuation valuation' loose
    (rewrite_arbitrary_u fx ctxu tenv canon cm reprs fuel).
Proof.
  intros Hrw m t w Hal Ht Hsat.
  unfold is_alias in Hal.
  destruct (get_alias_opt t) as [s | ] eqn:Ha; [ | discriminate ].
  pose proof (proj1 (get_alias_satisfied _ _ w t s Ha) Hsat) as [Hs Hkt].
  destruct (proj2 Ht s (kind t) w (get_alias_in_aliases t s Ha) Hs)
    as [(c & ->) | (sym & ->)].
  - apply (Hrw m t (Value_const c) (Value_const c) Ht);
      [ apply rel_u_of_const | exact Hsat ].
  - destruct fuel as [ | fuel]; cbn [rewrite_arbitrary_u];
      [ exact (sat_unknown_of_kind _ _ _ _ _ _ t Hsat eq_refl) | ].
    rewrite Ha.
    destruct (Typing_env.get_canonical_simple_exn tenv
                (Some Name_mode.in_types) None s)
      as [[[x | sym' | c] | ] | ] eqn:Hcan;
      try exact (sat_unknown_of_kind _ _ _ _ _ _ t Hsat eq_refl).
    + destruct (fix_alias fx && forgets m);
        [ exact (sat_unknown_of_kind _ _ _ _ _ _ t Hsat eq_refl) | ].
      apply alias_satisfied. split; [ | exact Hkt ].
      change (value_of_name valuation' (canon (Name_var x) (kind t) m)
              = Some (Value_symbol sym)).
      rewrite (Hcopies s (Name_var x) (kind t) m Hcan). exact Hs.
    + destruct (fix_alias fx && forgets m);
        [ exact (sat_unknown_of_kind _ _ _ _ _ _ t Hsat eq_refl) | ].
      apply alias_satisfied. split; [ | exact Hkt ].
      change (value_of_name valuation' (canon (Name_symbol sym') (kind t) m)
              = Some (Value_symbol sym)).
      rewrite (Hcopies s (Name_symbol sym') (kind t) m Hcan). exact Hs.
    + exfalso.
      pose proof (canonical_eval tenv statics valuation Henv s (Const c) Hcan)
        as Hceq.
      rewrite Hs in Hceq. discriminate Hceq.
Qed.
(* [closure_sound], with changes, of a closure that does not change. *)
Lemma closure_sound_u (rw : rewrite_metadata -> flambda_type -> flambda_type)
    (m : rewrite_metadata) (t : flambda_type) (v v' : value)
    (spec : closure_rewrite) :
  rw_sound_u rw -> rw_alias statics statics' valuation valuation' loose rw ->
  type_in_fragment_with loose valuation t -> unchanged_ext m v v' ->
  type_satisfied statics valuation v t ->
  rewrite_rule fx ctxu tenv m t = Rule_closure spec ->
  type_satisfied statics' valuation' v' (rewrite_closure fx tenv rw spec t).
Proof.
  intros Hrw Hrwa Ht Hv Hsat Hrule_u.
  destruct (rewrite_rule_u m t spec Hrule_u) as [Hrule Hunch].
  pose proof Hrule as Hrule0.
  unfold rewrite_rule in Hrule.
  destruct (forgets m) eqn:Hm; [ discriminate | ].
  destruct (kind t); try discriminate.
  destruct (Provers.meet_single_closures_entry tenv t)
    as [[[[[slot am] entry] ft] | | ] | ] eqn:Hmeet; try discriminate.
  destruct (meet_closures_head tenv t slot am entry ft Hmeet)
    as (e & known & isn & He & Hd & Hsingle).
  destruct (closure_of_type tenv statics valuation Henv t e known am isn slot
              entry v He Hd Hsingle Hsat)
    as [-> | (fdecls & vs & case & Hcis & Hl & Hent & Hcase)].
  { rewrite (rel_u_const m Null v' (or_introl Hv)).
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
  destruct (closure_rebuilt_view_u m v v' slot fdecls vs Hm Hv Hcis)
    as (sites & vs' & Hcis' & Hdesc & Hsites & Hslots & _ & Hproj).
  pose proof (closure_in_set_slot statics v slot fdecls vs Hcis) as Hslot_b.
  destruct (proj1 (is_bound_lookup slot fdecls) Hslot_b) as [id Hlslot].
  destruct (Hsets fdecls sites slot id Hlslot Hsites)
    as (S0 & HS0 & HS0in & HS0b).
  (* The allocation of the closure, which the metadata describes. *)
  assert (Hself : g_constructor G (sites slot) (Field_function_slot slot)
                    (sites slot)).
  { apply (proj2 (Hsites slot id Hlslot slot (sites slot))).
    split; [ exact Hslot_b | reflexivity ]. }
  (* Its set. *)
  set (nodes0 := map (fun p : function_slot * (node * Or_unknown.t code_id) =>
                        (fst p, fst (snd p))) S0).
  assert (Hnodes : ∀ (g : function_slot) (c : node),
             (g, c) ∈ nodes0 -> is_bound g fdecls = true ∧ c = sites g).
  { intros g c Hin. apply in_map_iff in Hin as ([g' [c' code]] & Hp & Hin).
    cbn [fst snd] in Hp. injection Hp as <- <-.
    exact (HS0in g' c' code Hin). }
  assert (HS0slot : is_bound slot S0 = true) by exact (HS0b slot Hslot_b).
  assert (Hat_g : ∀ g : function_slot,
             is_bound g fdecls = true -> closure_at S0 g = sites g).
  { intros g Hg. unfold closure_at.
    destruct (proj1 (is_bound_lookup g S0) (HS0b g Hg)) as [[c code] HlS0].
    rewrite HlS0.
    exact (proj2 (HS0in g c code (lookup_in g (c, code) S0 HlS0))). }
  pose proof (Hat_g slot Hslot_b) as Hat_slot.
  assert (Hcodes :
            ∀ p : function_slot * Or_unknown.t code_id,
              p ∈ map (fun g => (g, entry_code_id (Closures_entry fts cts vsts)
                                      g))
                    (domain cts) ->
              is_bound (fst p) S0 = true).
  { intros p Hp. apply in_map_iff in Hp as (g & <- & Hg). cbn [fst].
    apply HS0b. rewrite <- among_domain. apply Hidx_f.
    apply (proj1 (all_among _ _) Hcl). apply among_In. exact Hg. }
  destruct (closure_rule_inv fx is_local F T sets m slot am fts cts vsts _ spec
              Hrule)
    as (mbl0 & mbl & local_sets & first & rest & vsr & fsr
        & Hr & Hmem & Hf & Hspec).
  pose proof (refine_keeps fx Hfix_refine is_local F T sets Hleast m slot _
                mbl0 S0 (sites slot) Hm Hdesc Hself HS0 Hat_slot HS0slot
                Hcodes) as HS0loc.
  pose proof HS0loc as HS0ref.
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
    as (fsl & Hshape & Hfsl & Hfsl_md).
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
                ∧ s_has_usage sol from = true ∧ metadata_at sol mw from
                ∧ unchanged_at from).
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
    split; [ exact Hrv | split; [ exact Hhuf | split ] ].
    2:{ apply (unchanged_field (sites slot) from _
                 (proj2 (Hsh slot id Hlslot from) eq_refl)).
        intros g0 Habs. discriminate Habs. }
    intros a Ha Hha. apply Hdm.
    exact (follow_field_for_set_at is_local F T Hleast sets nodes0 w from m0
             (Hnodes_held w from Hsh) Hfol a Ha Hha). }
  (* The function slots of the rewritten type. *)
  assert (Hfs_kept :
            ∀ (g : function_slot) (p : rewrite_metadata * bool),
              lookup g fsr = Some p ->
              is_bound g cts = true ∧ is_bound g fdecls = true).
  { intros g p Hg. destruct (Hcov_f g p Hg) as (p0 & Hp0 & _).
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
        destruct (Hcta g tyg (lookup_in g tyg cts Htyg)) as [Hlo Hfrg].
        destruct (Hproj g closure' Hgd Hproj') as (c & Hpc & Hun & Hcu).
        pose proof (Hcts_sat g tyg c Htyg Hpc) as Hsc.
        destruct (is_alias tyg) eqn:Hal.
        * (* An alias: to a constant or a symbol, kept. *)
          unfold is_alias in Hal.
          destruct (get_alias_opt tyg) as [s | ] eqn:Ha; [ | discriminate ].
          destruct (proj1 (get_alias_satisfied _ _ c tyg s Ha) Hsc)
            as [Hs _].
          rewrite (Hun (proj2 Hfrg s (kind tyg) c
                          (get_alias_in_aliases tyg s Ha) Hs)).
          apply (Hrwa mg tyg c); [ | exact Hfrg | exact Hsc ].
          unfold is_alias. rewrite Ha. reflexivity.
        * (* Not an alias: the closure, rebuilt with its set, under the
             metadata of its slot, which describes its allocation. *)
          destruct Hlo as [Hlo | Habs]; [ | discriminate Habs ].
          assert (Hdg : describes sol mg (sites g)).
          { destruct (Hcov_f g (mg, erase) Hgf) as (pe & Hpe & Hdp).
            apply Hdp. rewrite (Hfsl_md g pe Hpe), (Hat_g g Hgd).
            reflexivity. }
          apply (Hrw mg tyg c closure' Hfrg); [ | exact Hsc ].
          left. right. right. exists sites, g.
          split; [ exact (Hcu Hlo mg Hdg) | ].
          destruct (proj1 (is_bound_lookup g S0) (HS0b g Hgd))
            as [[cg codeg] HlS0g].
          pose proof (lookup_in g (cg, codeg) S0 HlS0g) as HinS0g.
          rewrite <- (proj2 (HS0in g cg codeg HinS0g)).
          exact (Hunch slot am fts cts vsts ft mbl0 S0 g cg codeg eq_refl
                   HS0ref HinS0g).
      + intros w ty' contents Hw Hcont.
        rewrite lookup_map_keys in Hw.
        destruct (lookup w vsr) as [mw | ] eqn:Hwv; [ | discriminate Hw ].
        apply some_inj in Hw. cbn [fst snd] in Hw. subst ty'.
        destruct (Hvs_kept w mw Hwv)
          as (ty & x & from & x' & Hty & Hx & Hx' & Hrv & Hhuf & Hatf
              & Hunf).
        rewrite Hx' in Hcont. injection Hcont as <-.
        rewrite (destructure_env_var e known am isn slot fts cts vsts w Hd
                   Hsingle), Hty.
        apply (Hrw mw ty x x').
        * exact (Hvsf w ty (lookup_in w ty vsts Hty)).
        * left. right. left. exists from.
          split; [ exact Hrv | split; [ exact Hatf | ] ].
          split; [ intros _; exact Hhuf | exact Hunf ].
        * exact (Hvsts_sat w ty x Hty Hx). }
  destruct (fix_null fx && expanded_maybe_null tenv t); [ | exact Hty' ].
  unfold at_least_this_closure, create_closures, non_null_value in Hty' |- *.
  apply maybe_null_of_not_null. exact Hty'.
Qed.

(* The lookup of a map whose keys are renamed injectively. *)
Lemma lookup_rename {A B : Type} (h : positive -> positive)
    (a : positive * A -> B) (l : list (positive * A)) (k : positive) :
  (∀ g1 g2 : positive,
     g1 ∈ map fst l -> g2 ∈ map fst l -> h g1 = h g2 -> g1 = g2) ->
  k ∈ map fst l ->
  lookup (h k) (map (fun p => (h (fst p), a p)) l)
  = option_map (fun x => a (k, x)) (lookup k l).
Proof.
  induction l as [ | [k' x'] l IH]; intros Hinj Hk; [ destruct Hk | ].
  cbn [map fst lookup].
  destruct (Pos.eqb_spec k k') as [-> | Hne].
  - rewrite Pos.eqb_refl. reflexivity.
  - destruct Hk as [Hk | Hk]; [ cbn in Hk; congruence | ].
    destruct (Pos.eqb_spec (h k) (h k')) as [Heq | Hne'].
    + exfalso. apply Hne. apply Hinj;
        [ right; exact Hk | left; reflexivity | exact Heq ].
    + apply IH; [ | exact Hk ]. intros g1 g2 H1 H2. apply Hinj;
        right; assumption.
Qed.

Lemma lookup_nth {A : Type} (g : positive) (s : A)
    (l : list (positive * A)) :
  lookup g l = Some s -> ∃ i : nat, nth_error l i = Some (g, s).
Proof.
  induction l as [ | [k a] l IH]; cbn [lookup]; [ discriminate | ].
  destruct (Pos.eqb_spec g k) as [-> | _].
  - intros [= ->]. exists 0. reflexivity.
  - intros H. destruct (IH H) as [i Hi]. exists (S i). exact Hi.
Qed.

Lemma rename_decls_map (r : U.closure_repr)
    (decls : list (function_slot * code_id)) :
  U.rename_function_decls r decls
  = map (fun p => (U.rename_function_slot r (fst p), snd p)) decls.
Proof.
  unfold U.rename_function_decls. apply map_ext. intros [g id]. reflexivity.
Qed.

(* The renaming of the function slots is injective on those it renames. *)
Lemma rename_inj (r : U.closure_repr) (g1 g2 : function_slot) :
  NoDup (map snd (U.cr_function_slots r)) ->
  is_bound g1 (U.cr_function_slots r) = true ->
  is_bound g2 (U.cr_function_slots r) = true ->
  U.rename_function_slot r g1 = U.rename_function_slot r g2 -> g1 = g2.
Proof.
  intros Hnd H1 H2. unfold U.rename_function_slot.
  destruct (proj1 (is_bound_lookup _ _) H1) as [s1 Hl1].
  destruct (proj1 (is_bound_lookup _ _) H2) as [s2 Hl2].
  rewrite Hl1, Hl2. intros <-.
  destruct (lookup_nth g1 s1 _ Hl1) as [i1 Hi1].
  destruct (lookup_nth g2 s1 _ Hl2) as [i2 Hi2].
  assert (Hs1 : nth_error (map snd (U.cr_function_slots r)) i1 = Some s1)
    by exact (map_nth_error snd i1 _ Hi1).
  assert (Hs2 : nth_error (map snd (U.cr_function_slots r)) i2 = Some s1)
    by exact (map_nth_error snd i2 _ Hi2).
  assert (Hlt : i1 < length (map snd (U.cr_function_slots r))).
  { apply nth_error_Some. rewrite Hs1. discriminate. }
  pose proof (proj1 (NoDup_nth_error _) Hnd i1 i2 Hlt
                (eq_trans Hs1 (eq_sym Hs2))) as <-.
  rewrite Hi1 in Hi2. injection Hi2 as ->. reflexivity.
Qed.

Lemma same_names_of_in {A B : Type} (l1 : finite_map A) (l2 : finite_map B)
    :
  (∀ x : positive, x ∈ map fst l1 <-> x ∈ map fst l2) ->
  same_names (domain l1) (domain l2).
Proof.
  intros H x. unfold domain. rewrite !among_dedup.
  destruct (among x (map fst l1)) eqn:H1;
    destruct (among x (map fst l2)) eqn:H2; try reflexivity.
  - apply among_In in H1. apply H in H1. apply among_In in H1. congruence.
  - apply among_In in H2. apply H in H2. apply among_In in H2. congruence.
Qed.

Lemma bound_in_fst {A : Type} (g : positive) (l : list (positive * A)) :
  g ∈ map fst l -> is_bound g l = true.
Proof. intros H. rewrite is_bound_among. apply among_In. exact H. Qed.

Lemma fst_of_bound {A : Type} (g : positive) (l : list (positive * A)) :
  is_bound g l = true -> g ∈ map fst l.
Proof. intros H. rewrite is_bound_among in H. apply among_In. exact H. Qed.

(* The type that the rewriting gives an unknown type. *)
Lemma rewrite_u_unknown (fuel : nat) (m : rewrite_metadata)
    (k : flambda_kind) (w : value) :
  kind_of_value w = k ->
  type_satisfied statics' valuation' w
    (rewrite_arbitrary_u fx ctxu tenv canon cm reprs fuel m (unknown k)).
Proof.
  intros Hk.
  assert (Hu : type_satisfied statics' valuation' w (unknown k))
    by (apply unknown_satisfied; exact Hk).
  destruct fuel as [ | [ | fuel]]; cbn [rewrite_arbitrary_u rewrite_u];
    destruct k as [ | [] | | ]; try exact Hu;
    unfold changed_rule, rewrite_rule; destruct (forgets m); try exact Hu;
    cbn; try exact Hu.
Qed.

(* A value held in a field other than a function slot, related as the
   rebuild relates it there. *)
Lemma rel_u_held (m : rewrite_metadata) (a from : node) (f : field)
    (v v' : value) :
  g_constructor G a f from -> (∀ g : function_slot, f <> Field_function_slot g)
  -> rebuilt_value G sol v from v' -> metadata_at sol m from ->
  (forgets m = false -> s_has_usage sol from = true) -> rel_u m v v'.
Proof.
  intros Hcons Hf Hrv Hat Hhu. left. right. left. exists from.
  split; [ exact Hrv | split; [ exact Hat | split; [ exact Hhu | ] ] ].
  exact (unchanged_field a from f Hcons Hf).
Qed.

Lemma Forall2_map_l {A B C : Type} (Q : A -> C -> Prop) (R : B -> C -> Prop)
    (h : A -> B) :
  (∀ (a : A) (c : C), Q a c -> R (h a) c) ->
  ∀ (l : list A) (xs : list C), Forall2 Q l xs -> Forall2 R (map h l) xs.
Proof. intros Hqr l xs H. induction H; constructor; auto. Qed.

(* The new value slots of a closure whose set changes: the closure is
   [v] of the type [t], of the case [slot0] of the closures [known], the
   closures of its set [cl] being its allocations [sites]. *)
Section Slots.

Variable rw : rewrite_metadata -> flambda_type -> flambda_type.
Hypothesis Hrw : rw_sound_u rw.
Hypothesis Hrwu : ∀ (m0 : rewrite_metadata) (k : flambda_kind) (w : value),
  kind_of_value w = k ->
  type_satisfied statics' valuation' w (rw m0 (unknown k)).

Variable t : flambda_type.
Variable e0 : Expanded_type.t.
Variable known : finite_map row_like_closures_case.
Variable am0 : Alloc_mode.t.
Variable isn : is_null.
Variable slot0 : function_slot.
Variable id0 : code_id.
Variable fts : finite_map (Or_unknown.t function_type).
Variables cts vsts : finite_map flambda_type.
Variable fdecls : list (function_slot * code_id).
Variables vs vs_mid : list (value_slot * value).
Variable sites : function_slot -> node.
Variable cl : list node.

Hypothesis He : expand_head tenv t = Some e0.
Hypothesis Hd :
  Expanded_type.descr e0
  = Ok (Expanded_type.Head_value
          (Head_of_kind_value
             (Ok (Closures (Row_like_for_closures known) am0)) isn)).
Hypothesis Hsingle :
  single_closures_entry
    (row_like_for_closures_get_single_tag (Row_like_for_closures known))
  = Some (slot0, Closures_entry fts cts vsts).
Hypothesis Hvsf : ∀ (w : value_slot) (ty : flambda_type),
  (w, ty) ∈ vsts -> type_in_fragment_with loose valuation ty.
Hypothesis Hvb : ∀ (w : value_slot) (ty : flambda_type),
  lookup w vsts = Some ty -> is_bound w vs = true.
Hypothesis Hvsat : ∀ (w : value_slot) (ty : flambda_type) (x : value),
  lookup w vsts = Some ty -> lookup w vs = Some x ->
  type_satisfied statics valuation x ty.
Hypothesis Hl0 : lookup slot0 fdecls = Some id0.
Hypothesis Hsites : set_sites G fdecls sites.
Hypothesis Hslots :
  for_all_bindings
    (value_slot_rebuilt G sol (rebuilt_value G sol) fdecls sites vs_mid) vs.
Hypothesis Hcl : ∀ b : node,
  b ∈ cl <-> ∃ g : function_slot, is_bound g fdecls = true ∧ b = sites g.

Local Abbreviation reads := (U.get_fields_of_constructors F T is_local cl).

Definition slot_leaf_type (l : unboxed_leaf) : flambda_type :=
  let '(path, k, m) := l in
  match m with
  | None => unknown k
  | Some m => rw m (destructure_path tenv path t)
  end.

Definition slot_rel (pk : list field * flambda_kind) (x' : value) : Prop :=
  slot_value_at_path vs_mid (fst pk) = Some x' ∧ kind_of_value x' = snd pk.

Lemma site_in_cl : sites slot0 ∈ cl.
Proof.
  apply Hcl. exists slot0.
  split; [ unfold is_bound; rewrite Hl0; reflexivity | reflexivity ].
Qed.

(* A value slot read from the closures of the set is used: they hold it
   at one node, where the rebuild has it. *)
Lemma slot_read (w : value_slot) (use : U.field_use) (x : value) :
  (Field_value_slot w, use) ∈ reads -> lookup w vs = Some x ->
  ∃ (from : node) (x' : value),
    (∀ c : node, c ∈ cl -> g_constructor G c (Field_value_slot w) from)
    ∧ s_has_usage sol from = true
    ∧ lookup w vs_mid = Some x' ∧ rebuilt_value G sol x from x'.
Proof.
  intros Hin Hx.
  destruct (gfoc_used is_local F T sets Hleast cl _ use Hin)
    as (c & Hcin & Hfu).
  apply Hcl in Hcin as (g & Hg & ->).
  destruct (proj1 (is_bound_lookup g fdecls) Hg) as [idg Hlg].
  pose proof (proj1 (for_all_bindings_lookup _ _ _) Hslots w x Hx) as Hvr.
  destruct (Hvr g idg Hlg Hfu) as (from & x' & Hheld & _ & Hhu & Hlx & Hrv).
  exists from, x'.
  split; [ | split; [ exact Hhu | split; [ exact Hlx | exact Hrv ] ] ].
  intros c' Hc'. apply Hcl in Hc' as (g' & Hg' & ->).
  destruct (proj1 (is_bound_lookup g' fdecls) Hg') as [idg' Hlg'].
  exact (proj2 (Hheld g' idg' Hlg' from) eq_refl).
Qed.

Lemma slot_type (w : value_slot) :
  destructure_field (Field_value_slot w) (expand_head tenv t)
  = match lookup w vsts with
    | Some ty => ty
    | None => unknown Value
    end.
Proof.
  rewrite He. exact (destructure_value_slot e0 known am0 isn slot0 fts cts
                       vsts w Hd Hsingle).
Qed.

(* A new value slot that is the old one. *)
Lemma slot_leaf_sound (w : value_slot) (use : U.field_use) (x' : value) :
  (Field_value_slot w, use) ∈ reads ->
  lookup w vs_mid = Some x' -> kind_of_value x' = Value ->
  type_satisfied statics' valuation' x'
    (rw (pattern_metadata ctx
           (get_single_field_source ctx (sites slot0) (Field_value_slot w))
           use)
       (destructure_path tenv [Field_value_slot w] t)).
Proof.
  intros Hin Hx' Hk. cbn [destructure_path]. rewrite slot_type.
  destruct (lookup w vsts) as [ty | ] eqn:Hty;
    [ | exact (Hrwu _ Value x' Hk) ].
  destruct (proj1 (is_bound_lookup w vs) (Hvb w ty Hty)) as [x Hx].
  destruct (slot_read w use x Hin Hx) as (from & x'' & Hheld & Hhu & Hlx & Hrv).
  rewrite Hx' in Hlx. injection Hlx as <-.
  apply (Hrw _ ty x x' (Hvsf w ty (lookup_in _ _ _ Hty)));
    [ | exact (Hvsat w ty x Hty Hx) ].
  apply (rel_u_held _ (sites slot0) from (Field_value_slot w) x x'
           (Hheld _ site_in_cl));
    [ intros g Hg; discriminate Hg | exact Hrv | | intros _; exact Hhu ].
  exact (gfoc_metadata_at is_local F T sets Hleast cl (sites slot0) from _ use
           Hheld site_in_cl Hin).
Qed.

Lemma rw_held (m : rewrite_metadata) (t0 : flambda_type) (v v' : value) :
  type_in_fragment_with loose valuation t0 ->
  (∃ c : const, v = Value_const c ∧ v' = v)
  ∨ (∃ (a from : node) (f : field),
       g_constructor G a f from
       ∧ (∀ g : function_slot, f <> Field_function_slot g)
       ∧ rebuilt_value G sol v from v' ∧ metadata_at sol m from
       ∧ (forgets m = false -> s_has_usage sol from = true)) ->
  type_satisfied statics valuation v t0 ->
  type_satisfied statics' valuation' v' (rw m t0).
Proof.
  intros Ht0 [Hcst | (a & from & f & Hcons & Hf & Hrv & Hat & Hhu)] Hsat;
    apply (Hrw m t0 v v' Ht0); try exact Hsat.
  - left. left. exact Hcst.
  - exact (rel_u_held m a from f v v' Hcons Hf Hrv Hat Hhu).
Qed.

(* A value slot unboxed in turn. *)
Lemma slot_nested_sound (w : value_slot) (nested : U.unboxed_fields)
    (flow_to : list node) (s : node) (xs : list value) :
  (Field_value_slot w, U.Use_known flow_to) ∈ reads ->
  get_single_field_source ctx (sites slot0) (Field_value_slot w)
  = Field_one_source s ->
  (∃ (dominated : list (node * node)) (to_unbox : list node) (e : expr),
     cm = U.bound_code_my_closures e
     ∧ block_fields_of_one_kind G
     ∧ UCS.check_unboxed_uses F T
         (fst (U.unit_unboxed_fields F T dominated to_unbox e))
       = true
     ∧ forallb (U.has_to_be_unboxed dominated to_unbox) flow_to = true
     ∧ mk_shape F T dominated to_unbox e (get_direct_usages ctx flow_to)
         nested) ->
  Forall2 slot_rel
    (map (fun pk => (Field_value_slot w :: fst pk, snd pk))
       (unboxed_field_paths Value (U.Unboxed nested)))
    xs ->
  Forall2 (fun l x' => type_satisfied statics' valuation' x' (slot_leaf_type l))
    (map (fun l => let '(p, k, m) := l in (Field_value_slot w :: p, k, m))
       (unboxed_leaves ctx cm Value (get_direct_usages ctx flow_to) s
          (U.Unboxed nested)))
    xs.
Proof.
  intros Hin Hsrc (dominated & to_unbox & e & Hcm & Hkinds & Huses & Hall
                   & Hshape) Hp.
  pose proof (unboxed_leaves_paths ctx cm (U.Unboxed nested) Value
                (get_direct_usages ctx flow_to) s) as HLP.
  set (L := unboxed_leaves ctx cm Value (get_direct_usages ctx flow_to) s
              (U.Unboxed nested)) in *.
  set (P := unboxed_field_paths Value (U.Unboxed nested)) in *.
  destruct (lookup w vs_mid) as [y_mid | ] eqn:Hym.
  2: { destruct P as [ | pk P'] eqn:HP.
       - inversion Hp; subst. apply map_eq_nil in HLP. rewrite HLP.
         constructor.
       - inversion Hp as [ | pk0 x0 P0 xs0 [Hv _] _]; subst.
         cbn [fst slot_value_at_path] in Hv. rewrite Hym in Hv.
         discriminate Hv. }
  assert (HF : Forall2 (leaf_value_rel y_mid) P xs).
  { clear - Hp Hym. revert xs Hp. induction P as [ | pk P IH]; intros xs Hp;
      inversion Hp as [ | pk0 x0 P0 xs0 [Hv Hk] Hp']; subst; constructor.
    - cbn [fst slot_value_at_path] in Hv. rewrite Hym in Hv.
      split; [ exact Hv | exact Hk ].
    - exact (IH _ Hp'). }
  clear Hp.
  destruct (lookup w vs) as [y | ] eqn:Hy.
  - destruct (slot_read w _ y Hin Hy) as (from & y' & Hheld & Hhu & Hly & Hrv).
    rewrite Hym in Hly. injection Hly as <-.
    assert (Hty : ty_ok tenv statics valuation loose y
                    (destructure_field (Field_value_slot w)
                       (expand_head tenv t))).
    { rewrite slot_type. destruct (lookup w vsts) as [ty | ] eqn:Hvt.
      - apply (ty_ok_of tenv statics valuation loose Hfrag y ty
                 (Hvsf w ty (lookup_in _ _ _ Hvt))).
        left. exact (Hvsat w ty y Hvt Hy).
      - apply (ty_ok_of tenv statics valuation loose Hfrag y (unknown Value)
                 (unknown_fragment valuation Value)).
        right. exists Value. reflexivity. }
    pose proof (gfoc_level is_local F T sets Hleast dominated to_unbox e cl
                  (sites slot0) from s (Field_value_slot w) flow_to Hheld
                  site_in_cl Hhu Hsrc Hin Hall) as Hlevel.
    pose proof (leaves_sound_with is_local F T sets tenv statics statics'
                  valuation valuation' loose dominated to_unbox e Hleast Hlocal
                  Hkinds Henv Hfrag Huses rw (fun k => unknown k)
                  (fun k w' Hk => ltac:(apply unknown_satisfied; exact Hk))
                  rw_held (U.Unboxed nested)) as H.
    cbn beta iota in H.
    specialize (H Value (get_direct_usages ctx flow_to) s from y y_mid _ xs
                  Hlevel Hshape Hrv Hty HF).
    rewrite <- Hcm in H.
    refine (Forall2_map_l _ _ _ _ _ _ H).
    intros [[p k] [m | ]] x' Hl; exact Hl.
  - assert (Hvt : lookup w vsts = None).
    { destruct (lookup w vsts) as [ty | ] eqn:Hvt; [ | reflexivity ].
      pose proof (Hvb w ty Hvt) as Hb. unfold is_bound in Hb.
      rewrite Hy in Hb. discriminate Hb. }
    rewrite <- HLP in HF. apply Forall2_strip in HF. rewrite HLP in HF.
    refine (Forall2_map_l _ _ _ _ _ _ HF).
    intros [[p k] [m | ]] x' [Hmem [Hv Hk]]; cbn [slot_leaf_type].
    + cbn [destructure_path]. rewrite slot_type, Hvt.
      rewrite (destructure_path_unknown tenv (U.Unboxed nested) Value p k
                 y_mid x' Hmem Hv).
      exact (Hrwu m k x' Hk).
    + apply unknown_satisfied. exact Hk.
Qed.

(* The leaves with no variable. *)
Lemma slot_forgotten (f : field) (u : U.unboxed_field) (xs : list value) :
  Forall2 slot_rel
    (map (fun pk => (f :: fst pk, snd pk))
       (unboxed_field_paths (U.field_kind f) u))
    xs ->
  Forall2 (fun l x' => type_satisfied statics' valuation' x' (slot_leaf_type l))
    (forgotten_leaves f u) xs.
Proof.
  unfold forgotten_leaves. intros H.
  remember (map (fun pk => (f :: fst pk, snd pk))
              (unboxed_field_paths (U.field_kind f) u)) as P eqn:HP.
  assert (HPm : map (fun pk => (f :: fst pk, snd pk, @None rewrite_metadata))
                  (unboxed_field_paths (U.field_kind f) u)
                = map (fun pk => (fst pk, snd pk, None)) P).
  { rewrite HP, map_map. reflexivity. }
  rewrite HPm. clear HP HPm.
  induction H as [ | pk x' P xs [_ Hk] _ IH]; constructor; [ | exact IH ].
  cbn [slot_leaf_type]. apply unknown_satisfied. exact Hk.
Qed.

(* The fields of the representation, one by one. *)
Lemma slot_fields_sound (fg : bool) :
  ∀ (fields : U.unboxed_fields) (xs : list value),
    (∀ (f : field) (nested : U.unboxed_fields) (flow_to : list node),
       (f, U.Unboxed nested) ∈ fields ->
       (f, U.Use_known flow_to) ∈ reads ->
       ∃ (dominated : list (node * node)) (to_unbox : list node) (e : expr),
         cm = U.bound_code_my_closures e
         ∧ block_fields_of_one_kind G
         ∧ UCS.check_unboxed_uses F T
             (fst (U.unit_unboxed_fields F T dominated to_unbox e))
           = true
         ∧ forallb (U.has_to_be_unboxed dominated to_unbox) flow_to = true
         ∧ mk_shape F T dominated to_unbox e
             (get_direct_usages (plain_context is_local F T []) flow_to)
             nested) ->
    Forall2 slot_rel (unboxed_field_paths Value (U.Unboxed fields)) xs ->
    Forall2
      (fun l x' => type_satisfied statics' valuation' x' (slot_leaf_type l))
      (flat_map (changed_field_leaves ctx cm (sites slot0) reads fg) fields)
      xs.
Proof.
  induction fields as [ | [f u] fields IH]; intros xs Hnest Hp.
  { inversion Hp. constructor. }
  change (unboxed_field_paths Value (U.Unboxed ((f, u) :: fields)))
    with (map (fun pk => (f :: fst pk, snd pk))
            (unboxed_field_paths (U.field_kind f) u)
          ++ unboxed_field_paths Value (U.Unboxed fields)) in Hp.
  change (flat_map (changed_field_leaves ctx cm (sites slot0) reads fg)
            ((f, u) :: fields))
    with (changed_field_leaves ctx cm (sites slot0) reads fg (f, u)
          ++ flat_map (changed_field_leaves ctx cm (sites slot0) reads fg)
               fields).
  apply Forall2_app_inv_l in Hp as (xs1 & xs2 & Hp1 & Hp2 & ->).
  apply Forall2_app.
  2: { apply IH; [ | exact Hp2 ].
       intros f' n' ft H. exact (Hnest f' n' ft (or_intror H)). }
  pose proof (slot_forgotten f u xs1 Hp1) as Hforgot.
  unfold changed_field_leaves.
  destruct (Keyed.lookup field_eqb f reads) as [use | ] eqn:Hlk;
    [ | exact Hforgot ].
  apply (keyed_lookup_in field_eqb URP.field_eqb_decides) in Hlk.
  destruct fg; [ exact Hforgot | ]. cbv zeta.
  destruct u as [v | nested].
  - cbn [unboxed_field_paths map fst snd] in Hp1.
    inversion Hp1 as [ | pk x' P xs' [Hv Hk] Hnil]; subst.
    inversion Hnil; subst.
    destruct f as [ | w | | | | | | | ];
      cbn [fst slot_value_at_path] in Hv; try discriminate Hv.
    destruct (lookup w vs_mid) as [x0 | ] eqn:Hx; [ | discriminate Hv ].
    cbn [value_at_path] in Hv. injection Hv as ->.
    constructor; [ | constructor ].
    exact (slot_leaf_sound w use x' Hlk Hx Hk).
  - destruct use as [ | flow_to]; [ exact Hforgot | ].
    destruct (get_single_field_source ctx (sites slot0) f) as [ | s | ]
      eqn:Hsrc; try exact Hforgot.
    destruct xs1 as [ | x0 xs1].
    { inversion Hp1 as [Hnil | ]. symmetry in Hnil.
      apply map_eq_nil in Hnil.
      change (unboxed_field_paths (U.field_kind f) (U.Unboxed nested) = [])
        in Hnil.
      pose proof (unboxed_leaves_paths ctx cm (U.Unboxed nested)
                    (U.field_kind f) (get_direct_usages ctx flow_to) s)
        as Hstrip.
      rewrite Hnil in Hstrip. apply map_eq_nil in Hstrip.
      rewrite Hstrip. constructor. }
    destruct (unboxed_field_paths (U.field_kind f) (U.Unboxed nested))
      as [ | [p0 k0] P] eqn:Hpaths; [ inversion Hp1 | ].
    inversion Hp1 as [ | pk w' P' ws' [Hv0 _] _]; subst.
    destruct f as [ | w | | | | | | | ];
      cbn [fst slot_value_at_path] in Hv0; try discriminate Hv0.
    rewrite <- Hpaths in Hp1.
    exact (slot_nested_sound w nested flow_to s (x0 :: xs1) Hlk Hsrc
             (Hnest _ nested flow_to (or_introl eq_refl) Hlk) Hp1).
Qed.

(* The new value slots, of the leaves of the representation [r], and the
   value slots [vs'] of the closure of the third run. *)
Lemma changed_slots_sound (r : U.closure_repr)
    (vs' : list (value_slot * value)) :
  repr_leaves_ok is_local F T cm cl (U.cr_fields r) ->
  map fst vs' = map fst (U.fields_leaves (U.cr_fields r)) ->
  Forall2 slot_rel (UCS.fields_paths (U.cr_fields r)) (map snd vs') ->
  same_names
    (domain
       (map
          (fun q : unboxed_leaf * (variable * flambda_kind) =>
             let '((path, k, m), (slot', _)) := q in
             (slot',
              match m with
              | None => unknown k
              | Some m => rw m (destructure_path tenv path t)
              end))
          (combine (changed_leaves ctxu cm (sites slot0) reads (U.cr_fields r))
             (U.fields_leaves (U.cr_fields r)))))
    (domain vs')
  ∧ ∀ (w : value_slot) (ty : flambda_type) (x' : value),
      lookup w
        (map
           (fun q : unboxed_leaf * (variable * flambda_kind) =>
              let '((path, k, m), (slot', _)) := q in
              (slot',
               match m with
               | None => unknown k
               | Some m => rw m (destructure_path tenv path t)
               end))
           (combine
              (changed_leaves ctxu cm (sites slot0) reads (U.cr_fields r))
              (U.fields_leaves (U.cr_fields r))))
      = Some ty ->
      lookup w vs' = Some x' -> type_satisfied statics' valuation' x' ty.
Proof.
  intros [HR1 HR2] Hdom Hfor.
  set (fields := U.cr_fields r) in *.
  set (L := changed_leaves ctxu cm (sites slot0) reads fields).
  assert (Hvst :
            map
              (fun q : unboxed_leaf * (variable * flambda_kind) =>
                 let '((path, k, m), (slot', _)) := q in
                 (slot',
                  match m with
                  | None => unknown k
                  | Some m => rw m (destructure_path tenv path t)
                  end))
              (combine L (U.fields_leaves fields))
            = map (fun q => (fst (snd q), slot_leaf_type (fst q)))
                (combine L (U.fields_leaves fields))).
  { apply map_ext. intros [[[p k] m] [s kk]]. reflexivity. }
  rewrite Hvst.
  assert (HL : map (fun l => let '(p, k, _) := l in (p, k)) L
               = UCS.fields_paths fields
               ∧ Forall2
                   (fun l x' =>
                      type_satisfied statics' valuation' x' (slot_leaf_type l))
                   L (map snd vs')).
  { unfold L. rewrite changed_leaves_eq.
    destruct fields as [ | f0 fields0] eqn:Hf.
    - cbn. inversion Hfor. split; [ reflexivity | constructor ].
    - assert (Hne : classify_field_map
                      (filter (fun f => Keyed.is_bound field_eqb f reads)
                         (map fst (f0 :: fields0)))
                    <> Class_empty).
      { destruct (HR1 ltac:(discriminate)) as (f & Hf' & Hb).
        destruct (filter (fun f => Keyed.is_bound field_eqb f reads)
                    (map fst (f0 :: fields0)))
          as [ | f1 l1] eqn:Hfl; [ | apply classify_cons_not_empty ].
        exfalso. assert (Hin : f ∈ filter (fun f => Keyed.is_bound field_eqb
                                                       f reads)
                                   (map fst (f0 :: fields0)))
          by (apply filter_In; auto).
        rewrite Hfl in Hin. destruct Hin. }
      destruct (classify_field_map _) eqn:Hcls; [ contradiction | .. ];
        (split; [ apply changed_field_leaves_paths
                | apply slot_fields_sound; [ exact HR2 | exact Hfor ] ]). }
  destruct HL as [HLP HLF].
  assert (Hlen : length L = length (U.fields_leaves fields)).
  { rewrite (Forall2_length
               (FlambdaReaperUnboxingRefinesCallsSimProofs.fields_leaves_paths
                  fields)).
    rewrite <- HLP. symmetry. apply length_map. }
  destruct (combine_lookup (fun x' ty => type_satisfied statics' valuation' x'
                                           ty)
              slot_leaf_type L (U.fields_leaves fields) vs' Hlen Hdom HLF)
    as [Hkeys Hlk].
  split; [ | exact Hlk ].
  intros x. unfold domain. f_equal. f_equal. exact Hkeys.
Qed.

End Slots.

(* The rule of a closure whose set changes, applied. *)
Lemma changed_sound (rw : rewrite_metadata -> flambda_type -> flambda_type)
    (m : rewrite_metadata) (t : flambda_type) (v v' : value)
    (repr : U.closure_repr) (set : set_of_closures) (slot : function_slot)
    (am : Alloc_mode.t) (entry : closures_entry) :
  rw_sound_u rw ->
  (∀ (m0 : rewrite_metadata) (k : flambda_kind) (w : value),
     kind_of_value w = k ->
     type_satisfied statics' valuation' w (rw m0 (unknown k))) ->
  type_in_fragment_with loose valuation t -> rel_u m v v' ->
  type_satisfied statics valuation v t ->
  changed_rule fx ctxu tenv reprs m t = Some (repr, set, slot, am, entry) ->
  type_satisfied statics' valuation' v'
    (rewrite_changed fx ctxu tenv cm rw repr set slot am entry t).
Proof.
  intros Hrw Hrwu Ht Hv Hsat Hch.
  unfold changed_rule in Hch.
  destruct (forgets m) eqn:Hm; [ discriminate Hch | ].
  destruct (kind t) eqn:Hkt; try discriminate Hch.
  destruct (Provers.meet_single_closures_entry tenv t)
    as [[[[[slot0 am0] [fts cts vsts]] ft] | | ] | ] eqn:Hmeet;
    try discriminate Hch.
  cbv zeta in Hch.
  match type of Hch with
  | context [refine_sets_of_closures fx ctxu m slot0 ?cids ?mb] =>
    set (mb0 := mb) in Hch;
    destruct (refine_sets_of_closures fx ctxu m slot0 cids mb0)
      as [ls mbl] eqn:Hr
  end.
  destruct ls as [ | set0 [ | ? ?]]; destruct mbl; try discriminate Hch.
  destruct (Keyed.lookup node_eqb (closure_at set0 slot0) reprs)
    as [repr0 | ] eqn:Hrepr; [ | discriminate Hch ].
  injection Hch as <- <- <- <- <-.
  destruct (meet_closures_head tenv t slot0 am0 _ ft Hmeet)
    as (e & known & isn & He & Hd & Hsingle).
  destruct (closure_of_type tenv statics valuation Henv t e known am0 isn slot0
              _ v He Hd Hsingle Hsat)
    as [-> | (fdecls & vs & case & Hcis & Hl & Hent & Hcase)].
  { (* [Null] *)
    rewrite (rel_u_const m Null v' Hv).
    assert (Hmn : expanded_maybe_null tenv t = true).
    { unfold expanded_maybe_null. rewrite He, Hd.
      pose proof (expand_head_sound statics valuation tenv t e _ Henv Hsat He)
        as Hse.
      unfold Expanded_type.to_type in Hse. rewrite Hd in Hse.
      rewrite type_satisfied_value in Hse. cbn [descr_satisfied] in Hse.
      rewrite head_of_kind_value_satisfied_eq in Hse.
      destruct Hse as [[_ His] | [Hnn _]]; [ | contradiction ].
      destruct isn; [ contradiction | reflexivity ]. }
    unfold rewrite_changed. cbv zeta. rewrite Hfix_null, Hmn. cbn [andb].
    unfold exactly_this_closure, create_closures, non_null_value.
    rewrite type_satisfied_value. cbn [descr_satisfied].
    rewrite head_of_kind_value_satisfied_eq. left.
    split; [ reflexivity | exact I ]. }
  pose proof (expand_head_fragment valuation tenv Hfrag t e He Ht) as Hfr_e.
  unfold Expanded_type.to_type in Hfr_e. rewrite Hd in Hfr_e.
  destruct (closures_case_fragment valuation known am0 isn slot0 case fts cts
              vsts Hfr_e (lookup_in slot0 case known Hl) Hent)
    as (Hcl & Hvl & Hcta & Hvsf).
  destruct Hcase as (Hidx & Hentry & _).
  rewrite Hent in Hentry.
  apply closures_entry_lookup_iff in Hentry as (Hfts & Hcts_sat & Hvsts_sat).
  destruct (index_slots fdecls vs (case_index case) Hidx)
    as [Hidx_f Hidx_v].
  pose proof (closure_in_set_slot statics v slot0 fdecls vs Hcis) as Hslot_b.
  destruct (proj1 (is_bound_lookup slot0 fdecls) Hslot_b) as [id Hlslot].
  (* The set kept is the set of the closure. *)
  assert (Hkept :
            ∀ sites : function_slot -> node,
              set_sites G fdecls sites -> describes sol m (sites slot0) ->
              (∀ (g : function_slot) (c : node) (code : Or_unknown.t code_id),
                 (g, (c, code)) ∈ set0 ->
                 is_bound g fdecls = true ∧ c = sites g)
              ∧ (∀ g : function_slot,
                   is_bound g fdecls = true -> is_bound g set0 = true)).
  { intros sites Hsites Hdesc.
    destruct (Hsets fdecls sites slot0 id Hlslot Hsites)
      as (S0 & HS0 & HS0in & HS0b).
    assert (Hself : g_constructor G (sites slot0) (Field_function_slot slot0)
                      (sites slot0)).
    { apply (proj2 (Hsites slot0 id Hlslot slot0 (sites slot0))).
      split; [ exact Hslot_b | reflexivity ]. }
    assert (Hat_slot : closure_at S0 slot0 = sites slot0).
    { unfold closure_at.
      destruct (proj1 (is_bound_lookup slot0 S0) (HS0b slot0 Hslot_b))
        as [[c code] HlS0].
      rewrite HlS0.
      exact (proj2 (HS0in slot0 c code (lookup_in slot0 (c, code) S0 HlS0))).
    }
    assert (Hcodes :
              ∀ p : function_slot * Or_unknown.t code_id,
                p ∈ map (fun g => (g, entry_code_id
                                        (Closures_entry fts cts vsts) g))
                      (domain cts) ->
                is_bound (fst p) S0 = true).
    { intros p Hp. apply in_map_iff in Hp as (g & <- & Hg). cbn [fst].
      apply HS0b. rewrite <- among_domain. apply Hidx_f.
      apply (proj1 (all_among _ _) Hcl). apply among_In. exact Hg. }
    pose proof (refine_keeps fx Hfix_refine is_local F T sets Hleast m slot0
                  _ mb0 S0 (sites slot0) Hm Hdesc Hself HS0 Hat_slot
                  (HS0b slot0 Hslot_b) Hcodes) as Hk.
    assert (Hk' : S0 ∈ fst (refine_sets_of_closures fx ctxu m slot0
                              (map (fun g => (g, entry_code_id
                                                   (Closures_entry fts cts
                                                      vsts) g))
                                 (domain cts))
                              mb0)) by exact Hk.
    rewrite Hr in Hk'. destruct Hk' as [<- | []].
    split; [ exact HS0in | exact HS0b ]. }
  assert (Hat :
            ∀ sites : function_slot -> node,
              set_sites G fdecls sites -> describes sol m (sites slot0) ->
              closure_at set0 slot0 = sites slot0).
  { intros sites Hsites Hdesc.
    destruct (Hkept sites Hsites Hdesc) as [Hin Hb].
    unfold closure_at.
    destruct (proj1 (is_bound_lookup slot0 set0) (Hb slot0 Hslot_b))
      as [[c code] HlS0].
    rewrite HlS0.
    exact (proj2 (Hin slot0 c code (lookup_in slot0 (c, code) set0 HlS0))).
  }
  destruct Hv as [Hun | Hchg].
  { (* A closure that does not change: its set is not kept so. *)
    exfalso.
    destruct (closure_rebuilt_view_u m v v' slot0 fdecls vs Hm Hun Hcis)
      as (sites & vs' & _ & Hdesc & Hsites & _ & Hnone & _).
    rewrite (Hat sites Hsites Hdesc), Hnone in Hrepr. discriminate Hrepr. }
  destruct Hchg as (slot1 & fdecls1 & vs1 & vs_mid & sites & r & Hveq & Hb1
                    & Hsites1 & Hslots & Hdesc1 & Hlr & Hcr).
  subst v. cbn [closure_in_set] in Hcis.
  destruct (lookup slot1 fdecls1) as [id1 | ]; cbn in Hcis;
    [ | discriminate Hcis ].
  injection Hcis as -> -> ->.
  rewrite (Hat sites Hsites1 Hdesc1), Hlr in Hrepr.
  injection Hrepr as <-.
  destruct (Hkept sites Hsites1 Hdesc1) as [HS0in HS0b].
  destruct Hcr as (decls & vsA & vs' & Heq1 & Hv'eq & Hcur & Hdom & Hfor).
  injection Heq1 as Hcs <- <-.
  unfold U.current_slot in Hv'eq, Hcur. rewrite <- Hcs in Hv'eq, Hcur.
  destruct Hreprs as [Hfsp Hrp].
  pose proof (Hrp (sites slot0) r Hlr) as Hnd_fs.
  (* The closures of the set at the other slots. *)
  assert (Hrg :
            ∀ g : function_slot,
              is_bound g fdecls = true ->
              is_bound g (U.cr_function_slots r) = true
              ∧ ∃ rg : U.closure_repr,
                  Keyed.lookup node_eqb (sites g) reprs = Some rg
                  ∧ U.cr_fields rg = U.cr_fields r
                  ∧ U.cr_function_slots rg = U.cr_function_slots r
                  ∧ U.cr_current rg = g).
  { intros g Hg.
    pose proof (Hfsp (sites slot0) (sites g) g
                  (proj2 (Hsites1 slot0 id Hlslot g (sites g))
                     (conj Hg eq_refl))) as H.
    rewrite Hlr in H.
    destruct (Keyed.lookup node_eqb (sites g) reprs) as [rg | ];
      [ | contradiction ].
    destruct H as (H1 & H2 & H3 & H4).
    split; [ exact H4 | exists rg; auto ]. }
  assert (Hinj :
            ∀ g1 g2 : function_slot,
              g1 ∈ map fst fdecls -> g2 ∈ map fst fdecls ->
              U.rename_function_slot r g1 = U.rename_function_slot r g2 ->
              g1 = g2).
  { intros g1 g2 H1 H2.
    apply (rename_inj r g1 g2 Hnd_fs);
      [ exact (proj1 (Hrg g1 (bound_in_fst g1 fdecls H1)))
      | exact (proj1 (Hrg g2 (bound_in_fst g2 fdecls H2))) ]. }
  assert (Hlren :
            ∀ g : function_slot,
              g ∈ map fst fdecls ->
              lookup (U.rename_function_slot r g)
                (U.rename_function_decls r fdecls)
              = lookup g fdecls).
  { intros g Hg. rewrite rename_decls_map.
    pose proof (lookup_rename (U.rename_function_slot r) snd fdecls g Hinj Hg)
      as E.
    etransitivity; [ exact E | ]. destruct (lookup g fdecls); reflexivity. }
  assert (Hset_in :
            ∀ (g : function_slot) (c : node) (code : Or_unknown.t code_id),
              (g, (c, code)) ∈ set0 -> g ∈ map fst fdecls).
  { intros g c code Hin. apply fst_of_bound.
    exact (proj1 (HS0in g c code Hin)). }
  assert (Hinj0 :
            ∀ g1 g2 : function_slot,
              g1 ∈ map fst set0 -> g2 ∈ map fst set0 ->
              U.rename_function_slot r g1 = U.rename_function_slot r g2 ->
              g1 = g2).
  { intros g1 g2 H1 H2. apply Hinj.
    - apply in_map_iff in H1 as ([g' [c code]] & <- & H1).
      exact (Hset_in _ _ _ H1).
    - apply in_map_iff in H2 as ([g' [c code]] & <- & H2).
      exact (Hset_in _ _ _ H2). }
  (* The new value slots. *)
  set (cl := map (fun p : function_slot * (node * Or_unknown.t code_id) =>
                    fst (snd p))
               set0).
  assert (Hclm : ∀ b : node,
            b ∈ cl
            <-> ∃ g : function_slot, is_bound g fdecls = true ∧ b = sites g).
  { intros b. unfold cl. rewrite in_map_iff. split.
    - intros ([g [c code]] & <- & Hin). cbn [fst snd].
      destruct (HS0in g c code Hin) as [Hg ->]. exists g. auto.
    - intros (g & Hg & ->).
      destruct (proj1 (is_bound_lookup g set0) (HS0b g Hg))
        as [[c code] Hlg].
      exists (g, (c, code)). cbn [fst snd].
      split; [ exact (proj2 (HS0in g c code (lookup_in _ _ _ Hlg)))
             | exact (lookup_in _ _ _ Hlg) ]. }
  assert (Hvb : ∀ (w : value_slot) (ty : flambda_type),
            lookup w vsts = Some ty -> is_bound w vs = true).
  { intros w ty Hw. rewrite <- among_domain. apply Hidx_v.
    apply (proj1 (all_among _ _) Hvl). rewrite among_domain.
    apply is_bound_lookup. exists ty. exact Hw. }
  pose proof (Hleaves fdecls sites slot0 id r cl Hlslot Hsites1 Hlr Hclm)
    as Hrl.
  destruct (changed_slots_sound rw Hrw Hrwu t e known am0 isn slot0 id fts
              cts vsts fdecls vs vs_mid sites cl He Hd Hsingle Hvsf Hvb
              Hvsts_sat Hlslot Hslots Hclm r vs' Hrl Hdom Hfor)
    as [Hsn Hvs].
  unfold rewrite_changed. cbv zeta. rewrite He.
  rewrite (Hat sites Hsites1 Hdesc1).
  match goal with
  | |- context [exactly_this_closure ?s ?a ?b ?c ?d] =>
    assert (Hty' : type_satisfied statics' valuation' v'
                     (exactly_this_closure s a b c d))
  end.
  { apply exactly_this_closure_satisfied.
    exists (U.rename_function_decls r fdecls), vs'.
    split; [ rewrite Hv'eq; cbn [closure_in_set]; rewrite Hcur, Hlslot;
             reflexivity | ].
    split; [ | split ].
    - apply same_names_of_in. intros x. rewrite rename_decls_map, !map_map.
      cbn [fst]. split; intros Hx; apply in_map_iff in Hx as (p & <- & Hp);
        apply in_map_iff.
      + destruct p as [g [c code]].
        destruct (proj1 (is_bound_lookup g fdecls)
                    (proj1 (HS0in g c code Hp))) as [idg Hlg].
        exists (g, idg). split; [ reflexivity | exact (lookup_in _ _ _ Hlg) ].
      + destruct p as [g idg].
        assert (Hg : is_bound g fdecls = true).
        { apply bound_in_fst. apply in_map_iff. exists (g, idg). auto. }
        destruct (proj1 (is_bound_lookup g set0) (HS0b g Hg))
          as [[c code] Hlg].
        exists (g, (c, code)).
        split; [ reflexivity | exact (lookup_in _ _ _ Hlg) ].
    - exact Hsn.
    - apply closures_entry_lookup_iff. split; [ | split ].
      + intros g' ft' code Hg' Hcode.
        destruct (lookup_nth _ _ _ Hg') as [i Hi].
        rewrite nth_error_map in Hi.
        destruct (nth_error set0 i) as [[g [c oc]] | ] eqn:Hni;
          [ | discriminate Hi ].
        cbn in Hi. injection Hi as Hgg' Hft.
        assert (Hgin : g ∈ map fst set0).
        { apply in_map_iff. exists (g, (c, oc)).
          split; [ reflexivity | exact (nth_error_In _ _ Hni) ]. }
        subst g'.
        pose proof (lookup_rename (U.rename_function_slot r)
                   (fun p : function_slot * (node * Or_unknown.t code_id) =>
                      if closure_called ctxu (fst (snd p))
                      then
                        match entry_code_id (Closures_entry fts cts vsts)
                                (fst p)
                        with
                        | Or_unknown.Known code0 =>
                          Or_unknown.Known (Function_type code0)
                        | Or_unknown.Unknown => Or_unknown.Unknown
                        end
                      else Or_unknown.Unknown)
                   set0 g Hinj0 Hgin) as E.
        pose proof (eq_trans (eq_sym E) Hg') as Hg''. clear E Hg'.
        rename Hg'' into Hg'.
        rewrite (Hlren g (Hset_in g c oc (nth_error_In _ _ Hni))) in Hcode.
        destruct (lookup g set0) as [[c' oc'] | ]; cbn in Hg';
          [ | discriminate Hg' ].
        destruct (closure_called ctxu c'); [ | discriminate Hg' ].
        unfold entry_code_id in Hg'.
        cbn [closures_entry_find_function_type] in Hg'.
        destruct (lookup g fts) as [[ft1 | ] | ] eqn:Hlf; try discriminate Hg'.
        injection Hg' as <-. cbn [function_type_code_id].
        exact (Hfts g ft1 code Hlf Hcode).
      + intros g' ty closure Hg' Hproj.
        destruct (lookup_nth _ _ _ Hg') as [i Hi].
        rewrite nth_error_map in Hi.
        destruct (nth_error set0 i) as [[g [c oc]] | ] eqn:Hni;
          [ | discriminate Hi ].
        cbn in Hi. injection Hi as Hgg' _.
        assert (Hgin : g ∈ map fst set0).
        { apply in_map_iff. exists (g, (c, oc)).
          split; [ reflexivity | exact (nth_error_In _ _ Hni) ]. }
        subst g'.
        pose proof (Hset_in g c oc (nth_error_In _ _ Hni)) as Hgf.
        pose proof (bound_in_fst g fdecls Hgf) as Hgb.
        pose proof (lookup_rename (U.rename_function_slot r)
                   (fun p : function_slot * (node * Or_unknown.t code_id) =>
                      rw (Single_source (fst (snd p)))
                        (destructure_closure
                           (fun c0 => row_like_for_closures_get_closure c0
                                        (fst p))
                           (Some e)))
                   set0 g Hinj0 Hgin) as E.
        pose proof (eq_trans (eq_sym E) Hg') as Hg''. clear E Hg'.
        rename Hg'' into Hg'.
        destruct (lookup g set0) as [[c' oc'] | ] eqn:Hlg;
          cbn [option_map] in Hg'; [ | discriminate Hg' ].
        injection Hg' as <-.
        pose proof (proj2 (HS0in g c' oc' (lookup_in _ _ _ Hlg))) as ->.
        cbn [fst snd].
        match goal with
        | |- type_satisfied _ _ _ (rw _ ?X) =>
          assert (Edc : X = match lookup g cts with
                            | Some ty => ty
                            | None => unknown Value
                            end)
        end.
        { rewrite <- (destructure_closure_type e known am0 isn slot0 fts cts
                        vsts g Hd Hsingle).
          reflexivity. }
        rewrite Edc.
        (* The closure of the third run at the new slot. *)
        rewrite Hv'eq in Hproj. cbn [project_function_slot] in Hproj.
        rewrite Pos.eqb_refl, Hcur, Hlslot in Hproj. cbn in Hproj.
        destruct (lookup (U.rename_function_slot r g)
                    (U.rename_function_decls r fdecls)) eqn:Hlrg;
          cbn in Hproj; [ | discriminate Hproj ].
        injection Hproj as <-.
        destruct (lookup g cts) as [tyg | ] eqn:Htyg;
          [ | exact (Hrwu _ Value
                       (Value_closure_of_set (U.rename_function_slot r g)
                          (U.rename_function_decls r fdecls) vs')
                       eq_refl) ].
        destruct (proj1 (is_bound_lookup g fdecls) Hgb) as [idg Hlgd].
        assert (Hpc : project_function_slot statics slot0 g
                        (Value_closure_of_set slot0 fdecls vs)
                      = Some (Value_closure_of_set g fdecls vs)).
        { cbn [project_function_slot]. rewrite Pos.eqb_refl, Hlslot, Hlgd.
          reflexivity. }
        pose proof (Hcts_sat g tyg _ Htyg Hpc) as Hsc.
        destruct (Hcta g tyg (lookup_in g tyg cts Htyg)) as [_ Hfrg].
        apply (Hrw (Single_source (sites g)) tyg
                 (Value_closure_of_set g fdecls vs)
                 (Value_closure_of_set (U.rename_function_slot r g)
                    (U.rename_function_decls r fdecls) vs')
                 Hfrg); [ | exact Hsc ].
        destruct (Hrg g Hgb) as (_ & rg & Hlrg' & Hfl & Hfs & Hcurg).
        right. exists g, fdecls, vs, vs_mid, sites, rg.
        split; [ reflexivity | split; [ exact Hgb | ] ].
        split; [ exact Hsites1 | split; [ exact Hslots | ] ].
        split; [ reflexivity | split; [ exact Hlrg' | ] ].
        assert (Hren : U.rename_function_slot rg = U.rename_function_slot r).
        { unfold U.rename_function_slot. rewrite Hfs. reflexivity. }
        assert (Hrend : U.rename_function_decls rg fdecls
                        = U.rename_function_decls r fdecls).
        { rewrite !rename_decls_map, Hren. reflexivity. }
        exists fdecls, vs_mid, vs'.
        split; [ rewrite Hcurg; reflexivity | ].
        split; [ unfold U.current_slot; rewrite Hcurg, Hren, Hrend;
                 reflexivity | ].
        split; [ unfold U.current_slot; rewrite Hcurg, Hren, Hrend;
                 exact (Hlren g Hgf) | ].
        rewrite Hfl. split; [ exact Hdom | exact Hfor ].
      + exact Hvs. }
  destruct (fix_null fx && expanded_maybe_null tenv t); [ | exact Hty' ].
  unfold exactly_this_closure, create_closures, non_null_value in Hty' |- *.
  apply maybe_null_of_not_null. exact Hty'.
Qed.

(* The rule of a closure that keeps no set that changes does not apply
   to a closure whose set changes. *)
Lemma changed_not_closure_rule (m : rewrite_metadata) (t : flambda_type)
    (v v' : value) (spec : closure_rewrite) :
  changed_ext m v v' -> type_in_fragment_with loose valuation t ->
  type_satisfied statics valuation v t ->
  rewrite_rule fx ctxu tenv m t = Rule_closure spec -> False.
Proof.
  intros Hchg Ht Hsat Hrule.
  destruct (rewrite_rule_u m t spec Hrule) as [Hrule_p Hunch].
  unfold rewrite_rule in Hrule_p.
  destruct (forgets m) eqn:Hm; [ discriminate Hrule_p | ].
  destruct (kind t); try discriminate Hrule_p.
  destruct (Provers.meet_single_closures_entry tenv t)
    as [[[[[slot0 am0] [fts cts vsts]] ft] | | ] | ] eqn:Hmeet;
    try discriminate Hrule_p.
  destruct (meet_closures_head tenv t slot0 am0 _ ft Hmeet)
    as (e & known & isn & He & Hd & Hsingle).
  destruct Hchg as (slot1 & fdecls & vs & vs_mid & sites & r & -> & _
                    & Hsites & _ & Hdesc & Hlr & _).
  destruct (closure_of_type tenv statics valuation Henv t e known am0 isn slot0
              _ _ He Hd Hsingle Hsat)
    as [Habs | (fdecls' & vs' & case & Hcis & Hl & Hent & Hcase)];
    [ discriminate Habs | ].
  cbn [closure_in_set] in Hcis.
  destruct (lookup slot1 fdecls) as [id | ] eqn:Hlslot; cbn in Hcis;
    [ | discriminate Hcis ].
  injection Hcis as -> <- <-.
  pose proof (expand_head_fragment valuation tenv Hfrag t e He Ht) as Hfr_e.
  unfold Expanded_type.to_type in Hfr_e. rewrite Hd in Hfr_e.
  destruct (closures_case_fragment valuation known am0 isn slot0 case fts cts
              vsts Hfr_e (lookup_in slot0 case known Hl) Hent)
    as (Hcl & _ & _ & _).
  destruct Hcase as (Hidx & _).
  destruct (index_slots fdecls vs (case_index case) Hidx) as [Hidx_f _].
  assert (Hslot_b : is_bound slot0 fdecls = true)
    by (unfold is_bound; rewrite Hlslot; reflexivity).
  destruct (Hsets fdecls sites slot0 id Hlslot Hsites)
    as (S0 & HS0 & HS0in & HS0b).
  assert (Hself : g_constructor G (sites slot0) (Field_function_slot slot0)
                    (sites slot0)).
  { apply (proj2 (Hsites slot0 id Hlslot slot0 (sites slot0))).
    split; [ exact Hslot_b | reflexivity ]. }
  destruct (proj1 (is_bound_lookup slot0 S0) (HS0b slot0 Hslot_b))
    as [[c code] HlS0].
  pose proof (lookup_in slot0 (c, code) S0 HlS0) as HinS0.
  assert (Hc0 : c = sites slot0) by exact (proj2 (HS0in _ _ _ HinS0)).
  assert (Hat_slot : closure_at S0 slot0 = sites slot0).
  { unfold closure_at. rewrite HlS0. exact Hc0. }
  assert (Hcodes :
            ∀ p : function_slot * Or_unknown.t code_id,
              p ∈ map (fun g => (g, entry_code_id
                                      (Closures_entry fts cts vsts) g))
                    (domain cts) ->
              is_bound (fst p) S0 = true).
  { intros p Hp. apply in_map_iff in Hp as (g & <- & Hg). cbn [fst].
    apply HS0b. rewrite <- among_domain. apply Hidx_f.
    apply (proj1 (all_among _ _) Hcl). apply among_In. exact Hg. }
  pose proof (refine_keeps fx Hfix_refine is_local F T sets Hleast m slot0
                _ Must_be_local S0 (sites slot0) Hm Hdesc Hself HS0 Hat_slot
                (HS0b slot0 Hslot_b) Hcodes) as Hk.
  pose proof (Hunch slot0 am0 fts cts vsts ft Must_be_local S0 slot0 c code
                eq_refl Hk HinS0) as Hnone.
  rewrite Hc0, Hlr in Hnone. discriminate Hnone.
Qed.

(* The rewriting with changes, by induction on its fuel. *)
Lemma rewrite_sound_u (fuel : nat) :
  rw_sound_u (rewrite_arbitrary_u fx ctxu tenv canon cm reprs fuel)
  ∧ rw_sound_u (rewrite_u fx ctxu tenv canon cm reprs fuel).
Proof.
  induction fuel as [ | fuel [IHa IHr]]; split; intros m t v v' Ht Hv Hsat.
  - exact (sat_unknown_of_kind _ _ _ _ v v' t Hsat (rel_u_kind m v v' Hv)).
  - exact (sat_unknown_of_kind _ _ _ _ v v' t Hsat (rel_u_kind m v v' Hv)).
  - cbn [rewrite_arbitrary_u].
    destruct (get_alias_opt t) as [s | ] eqn:Ha;
      [ | exact (IHr m t v v' Ht Hv Hsat) ].
    pose proof (proj1 (get_alias_satisfied _ _ v t s Ha) Hsat) as [Hs Hkt].
    pose proof (proj2 Ht s (kind t) v (get_alias_in_aliases t s Ha) Hs)
      as Hunalloc.
    destruct (Typing_env.get_canonical_simple_exn tenv
                (Some Name_mode.in_types) None s)
      as [[[x | sym | c] | ] | ] eqn:Hcan;
      try exact (sat_unknown_of_kind _ _ _ _ v v' t Hsat
                   (rel_u_kind m v v' Hv)).
    + destruct (fix_alias fx && forgets m);
        [ exact (sat_unknown_of_kind _ _ _ _ v v' t Hsat
                   (rel_u_kind m v v' Hv)) | ].
      rewrite (rel_u_unallocated m v v' Hunalloc Hv).
      apply alias_satisfied. split; [ | exact Hkt ].
      change (value_of_name valuation' (canon (Name_var x) (kind t) m)
              = Some v).
      rewrite (Hcopies s (Name_var x) (kind t) m Hcan). exact Hs.
    + destruct (fix_alias fx && forgets m);
        [ exact (sat_unknown_of_kind _ _ _ _ v v' t Hsat
                   (rel_u_kind m v v' Hv)) | ].
      rewrite (rel_u_unallocated m v v' Hunalloc Hv).
      apply alias_satisfied. split; [ | exact Hkt ].
      change (value_of_name valuation' (canon (Name_symbol sym) (kind t) m)
              = Some v).
      rewrite (Hcopies s (Name_symbol sym) (kind t) m Hcan). exact Hs.
    + pose proof (canonical_eval tenv statics valuation Henv s (Const c) Hcan)
        as Hceq.
      rewrite Hs in Hceq. cbn in Hceq. injection Hceq as <-.
      apply (IHr m (type_for_const c) (Value_const c) v').
      * apply type_for_const_fragment.
      * exact Hv.
      * apply type_for_const_satisfied.
  - cbn [rewrite_u].
    destruct (changed_rule fx ctxu tenv reprs m t)
      as [[[[[repr set] slot] am] entry] | ] eqn:Hch.
    { exact (changed_sound (rewrite_arbitrary_u fx ctxu tenv canon cm reprs
                              fuel)
               m t v v' repr set slot am entry IHa
               (fun m0 k w Hk => rewrite_u_unknown fuel m0 k w Hk)
               Ht Hv Hsat Hch). }
    destruct (rewrite_rule fx ctxu tenv m t) as [ | | spec] eqn:Hrule.
    + assert (Hm : forgets m = false).
      { unfold rewrite_rule in Hrule.
        destruct (forgets m); [ discriminate Hrule | reflexivity ]. }
      exact (identity_sound_u (rewrite_arbitrary_u fx ctxu tenv canon cm reprs
                                 fuel)
               IHa (fields_forget_u fuel) m t v v' Hm Ht Hv Hsat).
    + exact (sat_unknown_of_kind _ _ _ _ v v' t Hsat (rel_u_kind m v v' Hv)).
    + destruct Hv as [Hun | Hchg].
      * exact (closure_sound_u (rewrite_arbitrary_u fx ctxu tenv canon cm
                                  reprs fuel)
                 m t v v' spec IHa (rewrite_arbitrary_u_alias fuel IHa) Ht Hun
                 Hsat Hrule).
      * destruct (changed_not_closure_rule m t v v' spec Hchg Ht Hsat Hrule).
Qed.

End Changed_sound.

(* A closure whose set changes: the relation of the induction. *)
Lemma changed_ext_of_runs (F : facts) (T : tables) (statics : static_env)
    (reprs : list (node * U.closure_repr)) (m : rewrite_metadata)
    (n : node) (v v_mid v' : value) (is_local : field -> bool) :
  closed is_local (graph_of_facts F) (solution_of_tables T) ->
  closure_in_set statics v <> None ->
  rebuilt_value (graph_of_facts F) (solution_of_tables T) v n v_mid ->
  metadata_at (solution_of_tables T) m n ->
  changed_at (solution_of_tables T) reprs n v_mid v' ->
  changed_ext F T reprs m v v'.
Proof.
  intros Hc Hcis Hrv Hat [(a & Ha & Hha) Hall].
  destruct (Hall a Ha Hha) as (r0 & _ & Hcr0).
  destruct v as [c | code vs0 | slot fdecls vs | s | tag fields | elements
                | c | l | callee args arity];
    cbn [rebuilt_value closure_in_set] in Hrv, Hcis; try contradiction.
  2:{ destruct Hrv as (-> & _).
      destruct Hcr0 as (d & a1 & b1 & Habs & _). discriminate Habs. }
  destruct (lookup slot fdecls) as [id | ] eqn:Hl; [ | contradiction ].
  destruct Hrv as (sites & vs_mid & -> & Hmcf & Hsites & Hslots).
  assert (Hb : is_bound slot fdecls = true)
    by (unfold is_bound; rewrite Hl; reflexivity).
  assert (Hhs : s_has_source (solution_of_tables T) (sites slot) = true).
  { apply (has_source_function_slot _ _ _ Hc _ _ (sites slot)
             (proj2 (Hsites slot id Hl slot (sites slot))
                (conj Hb eq_refl))
             eq_refl). }
  destruct (Hall (sites slot) Hmcf Hhs) as (r & Hlr & Hcr).
  exists slot, fdecls, vs, vs_mid, sites, r.
  split; [ reflexivity | split; [ exact Hb | ] ].
  split; [ exact Hsites | split; [ exact Hslots | ] ].
  split; [ exact (Hat _ Hmcf Hhs) | split; [ exact Hlr | exact Hcr ] ].
Qed.

Theorem rewrite_changed_type_sound : rewrite_changed_type_sound_statement.
Proof.
  intros is_local F T sets dominated to_unbox tcr e tenv canon statics
    statics' valuation valuation' fuel m t n v v_mid v' Hleast Hlocal Hsets
    Henv Hfrag Halloc Hst Hwhole Hcopies Hunst Hreprs Hnoslots Ht Hcis Hrv
    Hat Hch Hsat.
  assert (Hle : least_either is_local F T) by (left; exact Hleast).
  pose proof (least_closed_tables is_local F T Hle) as Hc.
  exact (proj1 (rewrite_sound_u all_fixes eq_refl (or_introl eq_refl) eq_refl
                  eq_refl is_local F T sets tenv canon statics statics'
                  valuation valuation' true _ _ Hc Hle Hlocal Hsets Henv Hfrag
                  Halloc Hst Hcopies (fun _ => Hwhole) Hunst Hreprs
                  (reprs_leaves_ok_of_no_slots is_local F T _ _ Hnoslots) fuel)
           m t v v' Ht
           (or_intror (changed_ext_of_runs F T statics _ m n v v_mid v'
                         is_local Hc Hcis Hrv Hat Hch))
           Hsat).
Qed.

Theorem rewrite_changed_type_sound_landed :
  rewrite_changed_type_sound_landed_statement.
Proof.
  intros is_local F T sets dominated to_unbox tcr e tenv canon statics
    statics' valuation valuation' fuel m t n v v_mid v' Hleast Hlocal Hsets
    Henv Hfrag Halloc Hst Hwhole Hcopies Hunst Hreprs Hnoslots Ht Hcis Hrv
    Hat Hch Hsat.
  assert (Hle : least_either is_local F T) by (left; exact Hleast).
  pose proof (least_closed_tables is_local F T Hle) as Hc.
  exact (proj1 (rewrite_sound_u landed_fixes eq_refl (or_intror eq_refl) eq_refl
                  eq_refl is_local F T sets tenv canon statics statics'
                  valuation valuation' true _ _ Hc Hle Hlocal Hsets Henv Hfrag
                  Halloc Hst Hcopies (fun _ => Hwhole) Hunst Hreprs
                  (reprs_leaves_ok_of_no_slots is_local F T _ _ Hnoslots) fuel)
           m t v v' Ht
           (or_intror (changed_ext_of_runs F T statics _ m n v v_mid v'
                         is_local Hc Hcis Hrv Hat Hch))
           Hsat).
Qed.

(* ======================================================================== *)
(* The representations of the unit                                          *)
(* ======================================================================== *)

(* The changes of representation from the table [tcr]: each name has the
   representation of the set of a closure [x] of the table, whose fields
   are those that the analysis makes of the fields read from the closures
   of that set, at the function slot of a closure of it that dominates the
   name. *)
Lemma changed_representations_in (F : facts) (T : tables)
    (cm : finite_map variable) (dominated : list (node * node))
    (to_unbox : list node) (is_local : field -> bool) (tcr : list node)
    (next : positive) (c : node) (r : U.closure_repr) :
  (c, r)
    ∈ fst (U.changed_representations F T cm dominated to_unbox is_local tcr
             next) ->
  ∃ (x f0 : node) (fuel : nat) (next' : positive),
    (U.cr_current r, f0) ∈ U.set_function_slots F x
    ∧ (f0, c) ∈ dominated
    ∧ U.cr_fields r
      = fst (U.mk_unboxed_fields F T cm dominated to_unbox fuel x
               (U.get_fields_of_constructors F T is_local
                  (map snd (U.set_function_slots F x)))
               next').
Proof.
  rewrite UFP.changed_representations_depth.
  match goal with
  | |- context [List.fold_left ?h tcr _] =>
    set (step := h)
  end.
  assert (Hgen :
    ∀ (l : list node) (acc : list (node * U.closure_repr))
      (seen : list node) (n : positive),
      (∀ p : node * U.closure_repr,
         p ∈ acc ->
         ∃ (x f0 : node) (fuel : nat) (next' : positive),
           (U.cr_current (snd p), f0) ∈ U.set_function_slots F x
           ∧ (f0, fst p) ∈ dominated
           ∧ U.cr_fields (snd p)
             = fst (U.mk_unboxed_fields F T cm dominated to_unbox fuel x
                      (U.get_fields_of_constructors F T is_local
                         (map snd (U.set_function_slots F x)))
                      next')) ->
      ∀ p : node * U.closure_repr,
        p ∈ fst (fst (List.fold_left step l (acc, seen, n))) ->
        ∃ (x f0 : node) (fuel : nat) (next' : positive),
          (U.cr_current (snd p), f0) ∈ U.set_function_slots F x
          ∧ (f0, fst p) ∈ dominated
          ∧ U.cr_fields (snd p)
            = fst (U.mk_unboxed_fields F T cm dominated to_unbox fuel x
                     (U.get_fields_of_constructors F T is_local
                        (map snd (U.set_function_slots F x)))
                     next')).
  { induction l as [ | x l IH]; intros acc seen n Hacc; [ exact Hacc | ].
    cbn [List.fold_left].
    destruct (step (acc, seen, n) x) as [[acc' seen'] n'] eqn:Hstep.
    apply IH. intros p Hp.
    unfold step in Hstep. cbv beta iota in Hstep.
    destruct (mem_node x seen);
      [ injection Hstep as <- _ _; exact (Hacc p Hp) | ].
    destruct (U.set_function_slots F x) as [ | [fs0 f00] l0] eqn:Hl;
      [ injection Hstep as <- _ _; exact (Hacc p Hp) | ].
    destruct (U.mk_unboxed_fields F T cm dominated to_unbox
                (UF.unboxing_depth T) x
                (U.get_fields_of_constructors F T is_local
                   (map snd ((fs0, f00) :: l0)))
                n)
      as [fields n1] eqn:Hmk.
    destruct (U.rename_slots (map fst ((fs0, f00) :: l0)) n1) as [fss n2].
    injection Hstep as <- _ _.
    apply in_app_or in Hp as [Hp | Hp]; [ exact (Hacc p Hp) | ].
    assert (Hpf : ∃ (fs : function_slot) (f : node),
              (fs, f) ∈ (fs0, f00) :: l0
              ∧ p ∈ flat_map
                      (fun '(a, c) =>
                         if node_eqb a f
                         then [(c, U.Closure_repr fields fss fs)]
                         else [])
                      dominated).
    { apply in_app_or in Hp as [Hp1 | Hp1].
      - exists fs0, f00. split; [ left; reflexivity | exact Hp1 ].
      - apply in_flat_map in Hp1 as ([fs f] & Hfs & Hp2). exists fs, f.
        split; [ right; exact Hfs | exact Hp2 ]. }
    clear Hp. destruct Hpf as (fs & f & Hfs & Hp).
    apply in_flat_map in Hp as ([a c'] & Hd & Hp3).
    destruct (node_eqb a f) eqn:Haf; [ | destruct Hp3 ].
    destruct Hp3 as [<- | []]. apply node_eqb_eq in Haf. subst a.
    exists x, f, (UF.unboxing_depth T), n.
    cbn [fst snd U.cr_current U.cr_fields].
    rewrite Hl.
    split; [ exact Hfs | split; [ exact Hd | rewrite Hmk; reflexivity ] ]. }
  destruct (List.fold_left step tcr ([], [], next)) as [[acc seen] n] eqn:Hf.
  intros Hin. cbn [fst] in Hin.
  assert (Hin' : (c, r) ∈ fst (fst (List.fold_left step tcr ([], [], next))))
    by (rewrite Hf; exact Hin).
  exact (Hgen tcr [] [] next (fun p H => match H with end) (c, r) Hin').
Qed.

Lemma set_function_slots_in (F : facts) (x : node) (g : function_slot)
    (b : node) :
  (g, b) ∈ U.set_function_slots F x ->
  g_constructor (graph_of_facts F) x (Field_function_slot g) b.
Proof.
  unfold U.set_function_slots. intros H. apply in_compact in H.
  apply lookup_in in H.
  apply in_flat_map in H as ([[base f] y] & Hin & H).
  destruct f; try destruct H.
  destruct (node_eqb base x) eqn:E; [ | destruct H ].
  destruct H as [Heq | []]. injection Heq as <- <-.
  apply node_eqb_eq in E. subst. exact Hin.
Qed.

Lemma set_function_slots_has (F : facts) (x : node) (g : function_slot)
    (b : node) :
  g_constructor (graph_of_facts F) x (Field_function_slot g) b ->
  ∃ b' : node, (g, b') ∈ U.set_function_slots F x.
Proof.
  intros H. unfold U.set_function_slots.
  match goal with
  | |- ∃ _, _ ∈ compact ?l => set (L := l)
  end.
  assert (Hg : (g, b) ∈ L).
  { unfold L. apply in_flat_map. exists (x, Field_function_slot g, b).
    split; [ exact H | ]. cbn. rewrite node_eqb_refl. left. reflexivity. }
  assert (Hb : is_bound g L = true).
  { rewrite is_bound_among. apply among_In. apply in_map_iff.
    exists (g, b). auto. }
  apply is_bound_lookup in Hb as [b' Hb']. exists b'.
  apply in_compact. exact Hb'.
Qed.

(* The representations of the unit ask what the leaves need: the fields of
   the representation of a closure are read from the closures of its set,
   the set of a closure of [tcr] that dominates it. *)
Lemma unit_reprs_leaves_ok (is_local : field -> bool) (F : facts)
    (T : tables) (dominated : list (node * node)) (to_unbox tcr : list node)
    (e : expr) :
  least_either is_local F T ->
  block_fields_of_one_kind (graph_of_facts F) ->
  function_slots_mutual (graph_of_facts F) ->
  U.check_dominated T dominated = true ->
  UCS.check_unboxed_uses F T
    (fst (U.unit_unboxed_fields F T dominated to_unbox e))
  = true ->
  reprs_leaves_ok is_local F T (U.bound_code_my_closures e)
    (U.unit_changed_representations is_local F T dominated to_unbox tcr e).
Proof.
  intros Hleast Hkinds Hmut Hdomc Huses fdecls sites slot id r cl Hl Hsites
    Hlr Hcl.
  pose proof (least_closed_tables is_local F T Hleast) as Hc.
  apply (keyed_lookup_in node_eqb node_eqb_decides) in Hlr.
  unfold U.unit_changed_representations in Hlr.
  destruct (changed_representations_in F T _ dominated to_unbox is_local tcr
              _ _ r Hlr)
    as (x & f0 & fuel & next' & Hfs & Hd & Hfields).
  set (a := sites slot) in *.
  assert (Hself : g_constructor (graph_of_facts F) a (Field_function_slot slot)
                    a).
  { apply (proj2 (Hsites slot id Hl slot a)).
    split; [ unfold is_bound; rewrite Hl; reflexivity | reflexivity ]. }
  destruct (proj1 (FlambdaReaperUnboxingProofs.check_dominated_sound T
                     dominated Hdomc f0 a)
              (proj2 (mem_pair_In f0 a dominated) Hd))
    as (Hsrc & Hhs & Hnm).
  assert (Hna : s_any_source (solution_of_tables T) a = false).
  { apply not_true_iff_false. intros H. apply Hnm. left. exact H. }
  pose proof (sources_constructor _ _ _ Hc a (Field_function_slot slot) a
                Hself Hna) as Haa.
  pose proof (has_source_function_slot _ _ _ Hc a (Field_function_slot slot) a
                Hself eq_refl) as Hha.
  assert (Hf0 : f0 = a).
  { destruct (node_eqb f0 a) eqn:E; [ apply node_eqb_eq in E; exact E | ].
    exfalso. apply Hnm. right. exists f0, a.
    split; [ exact Hsrc | split; [ exact Hhs | split; [ exact Haa | ] ] ].
    split; [ exact Hha | ]. intros Heq. rewrite Heq, node_eqb_refl in E.
    discriminate E. }
  subst f0.
  pose proof (set_function_slots_in F x _ a Hfs) as Hxa.
  destruct (Hmut x a _ Hxa) as (g' & Hax).
  destruct (proj1 (Hsites slot id Hl g' x) Hax) as [Hg' Hx].
  destruct (proj1 (is_bound_lookup g' fdecls) Hg') as [idg' Hlg'].
  (* The closures of the set of [x] are those of the set. *)
  assert (Hxcl : ∀ b : node, b ∈ map snd (U.set_function_slots F x) <-> b ∈ cl).
  { intros b. rewrite Hcl. split.
    - intros Hb. apply in_map_iff in Hb as ([g b0] & <- & Hb). cbn [snd].
      pose proof (set_function_slots_in F x g b0 Hb) as Hc'. rewrite Hx in Hc'.
      destruct (proj1 (Hsites g' idg' Hlg' g b0) Hc') as [Hg ->].
      exists g. auto.
    - intros (g & Hg & ->).
      assert (Hcg : g_constructor (graph_of_facts F) x (Field_function_slot g)
                      (sites g)).
      { rewrite Hx. apply (proj2 (Hsites g' idg' Hlg' g (sites g))). auto. }
      destruct (set_function_slots_has F x g _ Hcg) as (b' & Hb').
      pose proof (set_function_slots_in F x g b' Hb') as Hcb.
      rewrite Hx in Hcb.
      destruct (proj1 (Hsites g' idg' Hlg' g b') Hcb) as [_ Hbeq].
      apply in_map_iff. exists (g, b'). split; [ exact Hbeq | exact Hb' ]. }
  split.
  - intros Hne.
    assert (Hex : ∃ (f : field) (u : U.unboxed_field), (f, u) ∈ U.cr_fields r).
    { destruct (U.cr_fields r) as [ | [f u] fields]; [ contradiction | ].
      exists f, u. left. reflexivity. }
    destruct Hex as (f & u & Hin).
    assert (Hfin : f ∈ map fst (U.cr_fields r))
      by (apply in_map_iff; exists (f, u); auto).
    rewrite Hfields in Hin.
    destruct (URP.mk_unboxed_fields_in F T _ dominated to_unbox fuel x _ next'
                f u Hin)
      as (use & Huse).
    destruct (gfoc_ext is_local F T _ cl f use Hxcl Huse)
      as (use' & Huse').
    exists f. split; [ exact Hfin | ].
    rewrite (keyed_is_bound_among field_eqb).
    apply (keyed_among_in field field_eqb URP.field_eqb_decides).
    apply in_map_iff. exists (f, use'). auto.
  - intros f nested flow_to Hn Hft.
    rewrite Hfields in Hn.
    destruct (mk_unboxed_fields_nested F T _ dominated to_unbox fuel x _ next'
                f nested Hn)
      as (flow_to' & fuel' & s & next'' & Hft' & Hall & Hneq).
    destruct (gfoc_spec is_local F T [] cl f _ Hft)
      as (_ & _ & _ & Hys).
    destruct (gfoc_spec is_local F T [] _ f _ Hft')
      as (_ & _ & _ & Hys').
    assert (Hsame : ∀ y : node, y ∈ flow_to <-> y ∈ flow_to').
    { intros y. rewrite Hys, Hys'.
      split; intros (c & Hcin & Hy); exists c;
        (split; [ apply Hxcl; exact Hcin | exact Hy ]). }
    exists dominated, to_unbox, e.
    split; [ reflexivity | split; [ exact Hkinds | split; [ exact Huses | ] ] ].
    split.
    + rewrite forallb_forall in Hall |- *. intros y Hy. apply Hall.
      apply Hsame. exact Hy.
    + rewrite Hneq.
      destruct (nested_shape is_local F T [] dominated to_unbox e Huses
                  flow_to' fuel' s next'' Hall)
        as (fu & blk & nx & Heq).
      exists fu, blk, nx. rewrite Heq.
      rewrite (get_fields_ext F T
                 (get_direct_usages (plain_context is_local F T []) flow_to')
                 (get_direct_usages (plain_context is_local F T []) flow_to));
        [ reflexivity | ].
      apply mem_node_iff_eq. intros z. rewrite !get_direct_usages_iff.
      split; intros (y & Hy & H1 & H2); exists y;
        (split; [ apply Hsame; exact Hy | split; assumption ]).
Qed.

(* The closure types of a set whose representation changes, the
   representations keeping value slots. *)
Theorem rewrite_changed_type_sound_slots :
  rewrite_changed_type_sound_slots_statement.
Proof.
  intros is_local F T sets dominated to_unbox tcr e tenv canon statics
    statics' valuation valuation' fuel m t n v v_mid v' Hleast Hlocal Hsets
    Hkinds Hmut Henv Hfrag Halloc Hst Hwhole Hcopies Hdomc Huses Hunst Hreprs
    Ht Hcis Hrv Hat Hch Hsat.
  assert (Hle : least_either is_local F T) by (left; exact Hleast).
  pose proof (least_closed_tables is_local F T Hle) as Hc.
  exact (proj1 (rewrite_sound_u all_fixes eq_refl (or_introl eq_refl) eq_refl
                  eq_refl is_local F T sets tenv canon statics statics'
                  valuation valuation' true _ _ Hc Hle Hlocal Hsets Henv Hfrag
                  Halloc Hst Hcopies (fun _ => Hwhole) Hunst Hreprs
                  (unit_reprs_leaves_ok is_local F T dominated to_unbox tcr e
                     Hle Hkinds Hmut Hdomc Huses)
                  fuel)
           m t v v' Ht
           (or_intror (changed_ext_of_runs F T statics _ m n v v_mid v'
                         is_local Hc Hcis Hrv Hat Hch))
           Hsat).
Qed.

Theorem rewrite_changed_type_sound_slots_landed :
  rewrite_changed_type_sound_slots_landed_statement.
Proof.
  intros is_local F T sets dominated to_unbox tcr e tenv canon statics
    statics' valuation valuation' fuel m t n v v_mid v' Hleast Hlocal Hsets
    Hkinds Hmut Henv Hfrag Halloc Hst Hwhole Hcopies Hdomc Huses Hunst Hreprs
    Ht Hcis Hrv Hat Hch Hsat.
  assert (Hle : least_either is_local F T) by (left; exact Hleast).
  pose proof (least_closed_tables is_local F T Hle) as Hc.
  exact (proj1 (rewrite_sound_u landed_fixes eq_refl (or_intror eq_refl) eq_refl
                  eq_refl is_local F T sets tenv canon statics statics'
                  valuation valuation' true _ _ Hc Hle Hlocal Hsets Henv Hfrag
                  Halloc Hst Hcopies (fun _ => Hwhole) Hunst Hreprs
                  (unit_reprs_leaves_ok is_local F T dominated to_unbox tcr e
                     Hle Hkinds Hmut Hdomc Huses)
                  fuel)
           m t v v' Ht
           (or_intror (changed_ext_of_runs F T statics _ m n v v_mid v'
                         is_local Hc Hcis Hrv Hat Hch))
           Hsat).
Qed.

(* ======================================================================== *)
(* The fixes to land                                                        *)
(* ======================================================================== *)

Theorem rewrite_type_sound_landed : rewrite_type_sound_landed_statement.
Proof.
  intros is_local F T sets tenv canon statics statics' valuation valuation'
    fuel m t v v' Hleast Hlocal Hsets Henv Hfrag Halloc Hst Hcopies.
  assert (Hle : least_either is_local F T) by (left; exact Hleast).
  apply (rewrite_type_sound_in landed_fixes false landed_fixes_sound);
    try assumption.
  intros H. discriminate H.
Qed.

Theorem rewrite_symbol_type_sound_landed :
  rewrite_symbol_type_sound_landed_statement.
Proof.
  intros is_local F T sets tenv canon statics statics' valuation valuation' s
    allocated Hleast Hlocal Hsets Henv Hfrag Halloc Hst.
  assert (Hle : least_either is_local F T) by (left; exact Hleast).
  apply (rewrite_symbol_type_sound_in landed_fixes false landed_fixes_sound);
    try assumption.
  intros H. discriminate H.
Qed.

Theorem exported_types_sound_landed : exported_types_sound_landed_statement.
Proof. exact (exported_types_sound_gen landed_fixes landed_fixes_sound). Qed.

Theorem rewrite_result_type_sound_landed :
  rewrite_result_type_sound_landed_statement.
Proof.
  intros is_local F T sets tenv canon statics statics' valuation valuation'
    code_var type_var v v' Hleast Hlocal Hsets Henv Hfrag Halloc Hst.
  assert (Hle : least_either is_local F T) by (left; exact Hleast).
  apply (rewrite_result_type_sound_in landed_fixes false landed_fixes_sound
           is_local F T sets tenv canon statics statics' valuation
           valuation'); try assumption.
  intros H. discriminate H.
Qed.

Theorem rewrite_unboxed_result_type_sound_landed :
  rewrite_unboxed_result_type_sound_landed_statement.
Proof.
  intros is_local F T sets dominated to_unbox e tenv canon statics statics'
    valuation valuation' code_var type_var fields v v_mid ws Hleast Hlocal
    Hsets Hkinds Henv Hfrag Halloc Hst Hcopies Huses Hlk Hv Hrv Hws.
  assert (Hle : least_either is_local F T) by (left; exact Hleast).
  eapply (rewrite_unboxed_result_type_sound_in landed_fixes eq_refl
            (or_intror eq_refl) eq_refl eq_refl is_local F T sets tenv canon
            statics statics' valuation valuation' false);
    try eassumption.
  intros H. discriminate H.
Qed.

Theorem rewrite_keeps_null_landed : rewrite_keeps_null_landed_statement.
Proof.
  intros ctx tenv canon statics statics' valuation valuation' fuel m t Henv
    Hcopies Hsat.
  exact (proj1 (rewrite_keeps_null_fuel landed_fixes eq_refl ctx tenv canon
                  statics statics' valuation valuation' Henv Hcopies fuel)
           m t Hsat).
Qed.

Theorem exported_types_sound_end_to_end_landed :
  exported_types_sound_end_to_end_landed_statement.
Proof.
  exact (exported_types_sound_end_to_end_gen landed_fixes landed_fixes_sound).
Qed.

(* ======================================================================== *)
(* Subkinds                                                                 *)
(* ======================================================================== *)

(* What the reaper with types keeps of subkinds
   ([rewrite_kind_with_subkind_sound]), by induction on the subkind, its
   shapes and their fields together ([subkind_comb]). Under a metadata
   that erases, and at [Anything] under one of a single source or of
   usages, nothing is stated ([erase_sound]). At [Anything] under one of
   any usage, every allocation that the value may come from has any
   usage, so that a boxed number keeps its contents and no value is made
   poison ([any_usage_not_poison]). Otherwise, a field that the
   allocation does not use is poison and [follow_field] erases what is
   stated of it ([field_metadata], from [follow_field_unused]), and one
   that it uses is rebuilt under the metadata that [follow_field] gives
   it ([follow_field_at]); a boxed number whose contents have no use is
   poison, and [rewrite_boxed_number_kind] erases what is stated of it;
   a kept [Tagged_immediate] is of a constant, which the rebuild keeps.
   A symbol is itself, bound in the second run as [statics_rebuilt]
   says. *)

Scheme subkind_mind := Induction for subkind Sort Prop
  with constructor_shapes_mind := Induction for constructor_shapes Sort Prop
  with field_subkinds_mind := Induction for field_subkinds Sort Prop.
Combined Scheme subkind_comb from subkind_mind, constructor_shapes_mind,
  field_subkinds_mind.

(* The test of a field is [value_at] of what is stated of it. *)
Lemma fields_of_subkinds_at (statics : static_env) (nb : bool) (sk : subkind)
    (sks : field_subkinds) (v : value) (fields : list value) :
  fields_of_subkinds statics (Fields_cons nb sk sks) (v :: fields)
  = kind_eqb (kind_of_value v) Value && value_at statics (nb, sk) v
    && fields_of_subkinds statics sks fields.
Proof. reflexivity. Qed.

(* A value of what is stated is of [Anything], nullable as stated. *)
Lemma value_at_anything (statics : static_env) (nb : bool) (sk : subkind)
    (v : value) :
  value_at statics (nb, sk) v = true ->
  value_at statics (nb, Subkind_anything) v = true.
Proof.
  intros H. destruct v as [c | | | | | | | | ]; try reflexivity.
  destruct c; try reflexivity; try exact H.
  cbn in H |- *. apply andb_true_iff in H as [-> _]. reflexivity.
Qed.

(* A tagged immediate is a constant. *)
Lemma tagged_immediate_const (v : value) :
  is_tagged_immediate v = true -> ∃ c : const, v = Value_const c.
Proof. destruct v; try discriminate; eauto. Qed.


Lemma is_number_of_kind_boxable (bn : boxable_number) (c : const) :
  is_number_of_kind bn c = true -> boxable_number_of_const c = Some bn.
Proof. destruct bn, c; try discriminate; reflexivity. Qed.

(* Whether a metadata erases what is stated, in [rewrite_subkind]. *)
Definition erasing (m : rewrite_metadata) : bool :=
  match m with
  | No_source | At_least_one_source_no_usages => true
  | Single_source _ | Many_sources_any_usage | Many_sources_usages _ => false
  end.

(* The proof is for any relation [R] of the values of the two runs and
   static environments that have of the values what the statement reads:
   [rebuilt_value] and [statics_rebuilt], and the relation of the
   simulation of the rebuild (FlambdaReaperRebuildProofs.rel and
   [statics_rel]), with which the rebuild with types is shown to refine
   the unit. *)
Section Subkinds_sound.

Variable is_local : field -> bool.
Variable F : facts.
Variable T : tables.
Variable sets : list set_of_closures.
Variables statics statics' : static_env.

Hypothesis Hleast : least_solution is_local F T.
Hypothesis Hlocal : is_local_slots_only is_local.

Local Abbreviation G := (graph_of_facts F).
Local Abbreviation sol := (solution_of_tables T).
Local Abbreviation ctx := (plain_context is_local F T sets).

Variable R : value -> node -> value -> Prop.

(* The field [i] of a block of values whose allocation is [a]. *)
Definition field_R (a : node) (i : nat) (field field' : value) : Prop :=
  kind_of_value field = Value ->
  ∃ from : node,
    g_constructor G a (Field_block i Value) from
    ∧ (s_field_used sol a (Field_block i Value) = true ->
       s_has_usage sol from = true ∧ R field from field')
    ∧ (s_field_used sol a (Field_block i Value) = false ->
       field' = Value_const (Poison Value)).

Hypothesis HR_kind :
  ∀ (v : value) (n : node) (v' : value),
    R v n v' -> kind_of_value v' = kind_of_value v.
Hypothesis HR_null :
  ∀ (v : value) (n : node) (v' : value),
    R v n v' -> v' = Value_const Null -> v = Value_const Null.
Hypothesis HR_const :
  ∀ (c : const) (n : node) (v' : value),
    R (Value_const c) n v' -> v' = Value_const c.
Hypothesis HR_poison :
  ∀ (v : value) (n : node) (v' : value) (k : flambda_kind),
    metadata_at sol Many_sources_any_usage n -> R v n v' ->
    v' = Value_const (Poison k) -> v = Value_const (Poison k).
Hypothesis HR_symbol :
  ∀ (s : symbol) (n : node) (v' : value),
    R (Value_symbol s) n v' ->
    v' = Value_symbol s ∧ may_come_from sol (Node_symbol s) n
    ∧ (s_has_usage sol n = true -> s_has_usage sol (Node_symbol s) = true).
Hypothesis HR_block :
  ∀ (tag : Z) (fields : list value) (n : node) (v' : value),
    R (Value_block tag fields) n v' ->
    ∃ (a : node) (fields' : list value),
      may_come_from sol a n ∧ s_has_source sol a = true
      ∧ v' = Value_block tag fields'
      ∧ fields_rebuilt (field_R a) 0 fields fields'.
Hypothesis HR_boxed :
  ∀ (c : const) (n : node) (v' : value),
    R (Value_boxed_number c) n v' ->
    ∃ (a : node) (bn : boxable_number) (from : node),
      may_come_from sol a n ∧ s_has_source sol a = true
      ∧ is_number_of_kind bn c = true
      ∧ g_constructor G a (Field_boxed_number bn) from
      ∧ v' = (if s_field_used sol a (Field_boxed_number bn)
              then Value_boxed_number c
              else Value_const (Poison Value)).
Hypothesis HS_sourced :
  ∀ (s : symbol) (allocated : statically_allocated_value),
    lookup s (static_values statics) = Some allocated ->
    s_has_source sol (Node_symbol s) = true.
Hypothesis HS_boxed :
  ∀ (s : symbol) (c : const),
    lookup s (static_values statics)
    = Some (Static_boxed_number (Value_const c)) ->
    s_has_usage sol (Node_symbol s) = true ->
    lookup s (static_values statics')
    = Some (Static_boxed_number (Value_const c)).
Hypothesis HS_block :
  ∀ (s : symbol) (tag : Z) (fields : list value),
    lookup s (static_values statics) = Some (Static_block tag fields) ->
    s_has_usage sol (Node_symbol s) = true ->
    ∃ fields' : list value,
      lookup s (static_values statics') = Some (Static_block tag fields')
      ∧ fields_rebuilt (field_R (Node_symbol s)) 0 fields fields'.

Lemma subkinds_closed : closed is_local G sol.
Proof. exact (least_closed_tables is_local F T (or_introl Hleast)). Qed.

Lemma block_not_local (i : nat) (k : flambda_kind) :
  is_local (Field_block i k) = false.
Proof.
  destruct (is_local (Field_block i k)) eqn:E; [ | reflexivity ].
  destruct (Hlocal _ E) as [(w & Hw) | (g & Hg)]; discriminate.
Qed.

Lemma boxed_not_local (bn : boxable_number) :
  is_local (Field_boxed_number bn) = false.
Proof.
  destruct (is_local (Field_boxed_number bn)) eqn:E; [ | reflexivity ].
  destruct (Hlocal _ E) as [(w & Hw) | (g & Hg)]; discriminate.
Qed.

(* What is kept at [Anything], under a metadata of any usage. *)
Lemma anything_sound (nb : bool) (v : value) (n : node) (v' : value) :
  metadata_at sol Many_sources_any_usage n -> R v n v' ->
  value_at statics (nb, Subkind_anything) v = true ->
  value_at statics' (nb, Subkind_anything) v' = true.
Proof.
  intros Hmd Hv Hat. destruct v' as [c' | | | | | | | | ]; try reflexivity.
  destruct c'; try reflexivity.
  - rewrite (HR_null v n _ Hv eq_refl) in Hat. exact Hat.
  - match goal with
    | |- value_at _ _ (Value_const (Poison ?k)) = true =>
      rewrite (HR_poison v n _ k Hmd Hv eq_refl) in Hat
    end.
    exact Hat.
Qed.

Lemma tagged_sound (nb : bool) (v : value) (n : node) (v' : value) :
  R v n v' ->
  value_at statics (nb, Subkind_tagged_immediate) v = true ->
  value_at statics' (nb, Subkind_tagged_immediate) v' = true.
Proof.
  intros Hv Hat. destruct v as [c | | | | | | | | ];
    try (cbn in Hat; discriminate Hat).
  rewrite (HR_const c n v' Hv). destruct c; exact Hat.
Qed.

(* What is erased. *)
Lemma erase_sound (nb : bool) (sk : subkind) (v : value) (n : node)
    (v' : value) :
  R v n v' -> value_at statics (nb, sk) v = true ->
  value_at statics' (erase_subkind (nb, sk)) v' = true.
Proof. intros _ _. apply FlambdaSubkindProofs.value_at_any_value. Qed.

Lemma rewrite_subkind_erasing (m : rewrite_metadata) (nb : bool)
    (sk : subkind) :
  erasing m = true ->
  rewrite_subkind ctx m nb sk = erase_subkind (nb, sk).
Proof. destruct m, sk; try discriminate; reflexivity. Qed.

Lemma rewrite_field_subkinds_any (sks : field_subkinds) :
  ∀ i : nat,
    rewrite_field_subkinds ctx Many_sources_any_usage i sks = sks.
Proof.
  induction sks as [ | nb sk rest IH]; intros i; [ reflexivity | ].
  cbn [rewrite_field_subkinds].
  change (follow_field ctx Many_sources_any_usage (Field_block i Value))
    with Many_sources_any_usage.
  rewrite IH. destruct sk; reflexivity.
Qed.

Lemma rewrite_shapes_any (shapes : constructor_shapes) :
  rewrite_shapes ctx Many_sources_any_usage shapes = shapes.
Proof.
  induction shapes as [ | tag rest IH | tag sks rest IH]; cbn [rewrite_shapes];
    [ reflexivity | rewrite IH; reflexivity | ].
  rewrite IH, rewrite_field_subkinds_any. reflexivity.
Qed.

(* Under a metadata that does not erase, [rewrite_subkind] at the head. *)
Definition head_rewrite (m : rewrite_metadata) (nb : bool) (sk : subkind)
    : param_subkind :=
  match sk with
  | Subkind_anything =>
    match m with
    | Many_sources_any_usage => (nb, sk)
    | No_source | At_least_one_source_no_usages | Single_source _
    | Many_sources_usages _ =>
      erase_subkind (nb, sk)
    end
  | Subkind_tagged_immediate => (nb, sk)
  | Subkind_boxed_number bn => rewrite_boxed_number_kind ctx m (nb, sk) bn
  | Subkind_variant consts shapes =>
    (nb, Subkind_variant consts (rewrite_shapes ctx m shapes))
  end.

Lemma rewrite_subkind_head (m : rewrite_metadata) (nb : bool)
    (sk : subkind) :
  erasing m = false -> rewrite_subkind ctx m nb sk = head_rewrite m nb sk.
Proof.
  intros Hm. destruct m; try discriminate Hm; destruct sk; try reflexivity.
  cbn [rewrite_subkind head_rewrite]. rewrite rewrite_shapes_any.
  reflexivity.
Qed.

(* The metadata of a field of a block whose allocation the metadata
   describes: [At_least_one_source_no_usages] where the allocation does not
   use it, and one that holds where the field is held otherwise. *)
Lemma field_metadata (m : rewrite_metadata) (a from : node) (f : field) :
  erasing m = false -> describes sol m a -> g_constructor G a f from ->
  is_local f = false ->
  (s_field_used sol a f = false ->
   follow_field ctx m f = At_least_one_source_no_usages)
  ∧ (s_field_used sol a f = true ->
     erasing (follow_field ctx m f) = true
     ∨ metadata_at sol (follow_field ctx m f) from).
Proof.
  intros Hm Hd Hcons Hloc.
  destruct (forgets m) eqn:Hf.
  - destruct m as [ | source | | [ | u us] | ]; try discriminate Hm;
      try discriminate Hf.
    split; intros _; [ reflexivity | left; reflexivity ].
  - split; intros Hfu.
    + exact (follow_field_unused ctx is_local subkinds_closed m a from f Hd
               Hf Hcons Hloc Hfu).
    + right.
      exact (follow_field_at ctx is_local subkinds_closed m a from f Hd Hf
               Hcons Hloc Hfu).
Qed.

Definition sk_sound (sk : subkind) : Prop :=
  ∀ (m : rewrite_metadata) (nb : bool) (v : value) (n : node) (v' : value),
    erasing m = false -> metadata_at sol m n -> s_has_usage sol n = true ->
    R v n v' -> value_at statics (nb, sk) v = true ->
    value_at statics' (rewrite_subkind ctx m nb sk) v' = true.

Definition shapes_sound (shapes : constructor_shapes) : Prop :=
  ∀ (m : rewrite_metadata) (a : node) (tag : Z) (fields fields' : list value),
    erasing m = false -> describes sol m a ->
    fields_rebuilt (field_R a) 0 fields fields' ->
    block_of_shapes statics shapes tag fields = true ->
    block_of_shapes statics' (rewrite_shapes ctx m shapes) tag
      fields'
    = true.

Definition field_subkinds_sound (sks : field_subkinds) : Prop :=
  ∀ (m : rewrite_metadata) (a : node) (i : nat)
    (fields fields' : list value),
    erasing m = false -> describes sol m a ->
    fields_rebuilt (field_R a) i fields fields' ->
    fields_of_subkinds statics sks fields = true ->
    fields_of_subkinds statics' (rewrite_field_subkinds ctx m i sks)
      fields'
    = true.

(* The allocation that a value at [n] comes from, described. *)
Lemma describes_at (m : rewrite_metadata) (a n : node) :
  metadata_at sol m n -> may_come_from sol a n -> s_has_source sol a = true ->
  describes sol m a.
Proof. intros Hmd Ha Hhs. exact (Hmd a Ha Hhs). Qed.

(* A name whose metadata is of its usages has a use. *)
Lemma metadata_usages_used (x : node) (us : list node) :
  metadata_for_name ctx x = Many_sources_usages us -> s_has_usage sol x = true.
Proof.
  unfold metadata_for_name. intros H.
  destruct (get_single_source ctx x);
    try (destruct (has_use ctx x); discriminate H).
  unfold get_usages in H.
  destruct (has_use ctx x) eqn:Hh; [ exact Hh | discriminate H ].
Qed.

Lemma boxed_number_contents (S : static_env) (nb : bool)
    (bn bn' : boxable_number) (c : const) :
  is_number_of_kind bn' c = true ->
  value_at S (nb, Subkind_boxed_number bn) (Value_boxed_number c) = true ->
  bn' = bn.
Proof.
  intros Hk Hat. pose proof (is_number_of_kind_boxable bn' c Hk) as Hb.
  unfold value_at in Hat. cbn [snd value_of_subkind boxed_number_of] in Hat.
  unfold boxable_contents in Hat.
  repeat (rewrite Hb in Hat; cbv beta iota in Hat).
  unfold boxable_number_eqb in Hat.
  destruct (boxable_number_eq_dec bn bn') as [-> | ]; [ reflexivity | ].
  discriminate Hat.
Qed.

Lemma subkinds_sound_all :
  (∀ sk : subkind, sk_sound sk)
  ∧ (∀ shapes : constructor_shapes, shapes_sound shapes)
  ∧ (∀ sks : field_subkinds, field_subkinds_sound sks).
Proof.
  apply subkind_comb.
  - (* Anything *)
    intros m nb v n v' Hm Hmd _ Hv Hat.
    rewrite rewrite_subkind_head by exact Hm. cbn [head_rewrite].
    destruct m; try apply FlambdaSubkindProofs.value_at_any_value.
    exact (anything_sound nb v n v' Hmd Hv Hat).
  - (* Tagged_immediate *)
    intros m nb v n v' Hm _ _ Hv Hat. rewrite rewrite_subkind_head by exact Hm.
    exact (tagged_sound nb v n v' Hv Hat).
  - (* A boxed number *)
    intros bn m nb v n v' Hm Hmd Hhu Hv Hat.
    rewrite rewrite_subkind_head by exact Hm. cbn [head_rewrite].
    unfold rewrite_boxed_number_kind.
    destruct (follow_field ctx m (Field_boxed_number bn)) eqn:Hff;
      try exact (erase_sound nb _ v n v' Hv Hat).
    all: destruct v as [c | | | s | tag fields | | c | | ];
      try (cbn in Hat; discriminate Hat).
    all: first
      [ (* A constant. *)
        solve
          [ rewrite (HR_const c n v' Hv);
            destruct c; try exact Hat; cbn in Hat; unfold says_nothing in Hat;
            rewrite andb_false_r in Hat; discriminate Hat ]
      | (* A symbol bound to a boxed number. *)
        solve
          [ destruct (HR_symbol _ _ _ Hv) as (-> & Hmay & Hsu);
            cbn [value_at fst snd value_of_subkind boxed_number_of] in Hat;
            cbn [value_at fst snd value_of_subkind
                 boxed_number_of];
            destruct (lookup s (static_values statics)) as [allocated | ]
              eqn:Hl; [ | discriminate Hat ];
            destruct allocated as [ | | | contents | | | ];
              try discriminate Hat;
            destruct contents as [c0 | | | | | | | | ]; try discriminate Hat;
            rewrite (HS_boxed s c0 Hl (Hsu Hhu)); exact Hat ]
      | (* A boxed number. *)
        solve
          [ destruct (HR_boxed _ _ _ Hv)
              as (a & bn' & from & Hmay & Hhs & Hk & Hcons & ->);
            rewrite (boxed_number_contents statics nb bn bn' c Hk Hat)
              in Hcons |- *;
            destruct (field_metadata m a from (Field_boxed_number bn) Hm
                        (Hmd a Hmay Hhs) Hcons (boxed_not_local bn))
              as [Hun _];
            destruct (s_field_used sol a (Field_boxed_number bn)) eqn:Hfu;
              [ exact Hat
              | rewrite (Hun eq_refl) in Hff; discriminate Hff ] ] ].
  - (* A variant *)
    intros consts shapes IH m nb v n v' Hm Hmd Hhu Hv Hat.
    rewrite rewrite_subkind_head by exact Hm. cbn [head_rewrite].
    destruct v as [c | | | s | tag fields | | c | | ];
      try (cbn in Hat; discriminate Hat).
    + rewrite (HR_const c n v' Hv).
      destruct c; try exact Hat; cbn in Hat; unfold says_nothing in Hat;
        rewrite andb_false_r in Hat; discriminate Hat.
    + destruct (HR_symbol _ _ _ Hv) as (-> & Hmay & Hsu).
      cbn [value_at fst snd value_of_subkind block_of] in Hat.
      cbn [value_at fst snd value_of_subkind block_of].
      destruct (lookup s (static_values statics)) as [allocated | ] eqn:Hl;
        [ | discriminate Hat ].
      destruct allocated as [ | | tag fields | | | | ]; try discriminate Hat.
      destruct (HS_block s tag fields Hl (Hsu Hhu))
        as (fields' & Hfound & Hfr).
      rewrite Hfound.
      apply (IH m (Node_symbol s) tag fields fields' Hm);
        [ | exact Hfr | exact Hat ].
      exact (Hmd (Node_symbol s) Hmay (HS_sourced s _ Hl)).
    + destruct (HR_block _ _ _ _ Hv)
        as (a & fields' & Hmay & Hhs & -> & Hfr).
      exact (IH m a tag fields fields' Hm (Hmd a Hmay Hhs) Hfr Hat).
  - (* No shape *)
    intros m a tag fields fields' _ _ _ Hat. discriminate Hat.
  - (* An undetermined shape *)
    intros tag' rest IH m a tag fields fields' Hm Hd Hfr Hat.
    cbn [rewrite_shapes block_of_shapes block_of_shapes] in Hat |- *.
    destruct (Z.eqb tag tag'); [ reflexivity | ].
    exact (IH m a tag fields fields' Hm Hd Hfr Hat).
  - (* A determined shape *)
    intros tag' sks IHsks rest IH m a tag fields fields' Hm Hd Hfr Hat.
    cbn [rewrite_shapes block_of_shapes block_of_shapes] in Hat |- *.
    destruct (Z.eqb tag tag').
    + exact (IHsks m a 0 fields fields' Hm Hd Hfr Hat).
    + exact (IH m a tag fields fields' Hm Hd Hfr Hat).
  - (* No field *)
    intros m a i fields fields' _ _ Hfr Hat.
    destruct fields as [ | field fields]; [ | discriminate Hat ].
    destruct fields' as [ | field' fields']; [ reflexivity | contradiction ].
  - (* A field *)
    intros nb sk IHsk rest IH m a i fields fields' Hm Hd Hfr Hat.
    destruct fields as [ | field fields]; [ discriminate Hat | ].
    destruct fields' as [ | field' fields']; [ contradiction | ].
    destruct Hfr as [Hfield Hfr].
    rewrite fields_of_subkinds_at in Hat.
    apply andb_true_iff in Hat as [Hat Hrest].
    apply andb_true_iff in Hat as [Hk Hat].
    cbn [rewrite_field_subkinds]. rewrite fields_of_subkinds_at.
    rewrite (IH m a (S i) fields fields' Hm Hd Hfr Hrest), andb_true_r.
    apply FlambdaPrimitiveProofs.kind_eqb_eq in Hk.
    destruct (Hfield Hk) as (from & Hcons & Hused & Hunused).
    assert (Hk' : kind_of_value field' = Value).
    { destruct (s_field_used sol a (Field_block i Value)).
      - rewrite (HR_kind _ _ _ (proj2 (Hused eq_refl))). exact Hk.
      - rewrite (Hunused eq_refl). reflexivity. }
    rewrite Hk', FlambdaPrimitiveProofs.kind_eqb_refl. cbn [andb].
    destruct (field_metadata m a from (Field_block i Value) Hm Hd Hcons
                (block_not_local i Value))
      as [Hun Hus].
    set (ps := rewrite_subkind ctx (follow_field ctx m (Field_block i Value))
                 nb sk).
    enough (Hps : value_at statics' ps field' = true)
      by (destruct ps; exact Hps).
    unfold ps.
    destruct (s_field_used sol a (Field_block i Value)) eqn:Hfu.
    + destruct (Hused eq_refl) as [Hhu Hv].
      destruct (erasing (follow_field ctx m (Field_block i Value))) eqn:He.
      * rewrite (rewrite_subkind_erasing _ nb sk He).
        exact (erase_sound nb sk field from field' Hv Hat).
      * destruct (Hus eq_refl) as [He' | Hmd]; [ congruence | ].
        exact (IHsk _ nb field from field' He Hmd Hhu Hv Hat).
    + rewrite (Hunused eq_refl), (Hun eq_refl),
        (rewrite_subkind_erasing At_least_one_source_no_usages nb sk eq_refl).
      apply FlambdaSubkindProofs.value_at_any_value.
Qed.

(* The statement, at the node of the name. *)
Lemma subkind_sound_at (x : node) (nb : bool) (sk : subkind) (v v' : value) :
  R v x v' -> value_at statics (nb, sk) v = true ->
  value_at statics'
    (rewrite_kind_with_subkind ctx x (nb, sk)) v'
  = true.
Proof.
  intros Hv Hat. unfold rewrite_kind_with_subkind. cbn [fst snd].
  destruct (metadata_for_name_at ctx x)
    as [Hmd Hhu].
  destruct (erasing (metadata_for_name ctx x)) eqn:He.
  - rewrite (rewrite_subkind_erasing _ nb sk He).
    exact (erase_sound nb sk v x v' Hv Hat).
  - apply (proj1 subkinds_sound_all sk _ nb v x v' He Hmd);
      [ | exact Hv | exact Hat ].
    destruct (forgets (metadata_for_name ctx x)) eqn:Hf;
      [ | exact (Hhu eq_refl) ].
    destruct (metadata_for_name ctx x) as [ | | | us | ] eqn:Hm;
      try discriminate He; try discriminate Hf.
    exact (metadata_usages_used x us Hm).
Qed.

End Subkinds_sound.

(* The relation [rebuilt_value]. *)
Section Subkinds_rebuilt.

Variable is_local : field -> bool.
Variable F : facts.
Variable T : tables.

Hypothesis Hleast : least_solution is_local F T.
Hypothesis Hlocal : is_local_slots_only is_local.

Local Abbreviation G := (graph_of_facts F).
Local Abbreviation sol := (solution_of_tables T).

Lemma rebuilt_null (v : value) (n : node) (v' : value) :
  rebuilt_value G sol v n v' -> v' = Value_const Null ->
  v = Value_const Null.
Proof.
  intros Hv Heq. subst v'.
  destruct v; cbn [rebuilt_value] in Hv; try contradiction.
  - destruct Hv as [_ Heq]. exact (eq_sym Heq).
  - destruct Hv as (sites & vs & Heq & _). discriminate Heq.
  - destruct Hv as [Heq _]. discriminate Heq.
  - destruct Hv as (a & fields' & _ & _ & Heq & _). discriminate Heq.
  - destruct Hv as (elements' & Heq). discriminate Heq.
  - destruct Hv as (a & bn & from & _ & _ & _ & _ & Heq).
    destruct (s_field_used sol a (Field_boxed_number bn)); discriminate Heq.
  - destruct Hv as (closure' & args' & Heq). discriminate Heq.
Qed.

Lemma rebuilt_const (c : const) (n : node) (v' : value) :
  rebuilt_value G sol (Value_const c) n v' -> v' = Value_const c.
Proof. intros [_ ->]. reflexivity. Qed.

(* What is kept at [Anything] and at [Tagged_immediate]. *)
(* Under a metadata of any usage, the rebuild makes poison of no value:
   the contents of a boxed number have a use. *)
Lemma any_usage_not_poison (v : value) (n : node) (v' : value)
    (k : flambda_kind) :
  metadata_at sol Many_sources_any_usage n -> rebuilt_value G sol v n v' ->
  v' = Value_const (Poison k) -> v = Value_const (Poison k).
Proof.
  intros Hmd Hv Heq0. subst v'.
  destruct v; cbn [rebuilt_value] in Hv; try contradiction.
  - destruct Hv as [_ Heq]. exact (eq_sym Heq).
  - destruct Hv as (sites & vs & Heq & _). discriminate Heq.
  - destruct Hv as [Heq _]. discriminate Heq.
  - destruct Hv as (a & fields' & _ & _ & Heq & _). discriminate Heq.
  - destruct Hv as (elements' & Heq). discriminate Heq.
  - destruct Hv as (a & bn & from & Hmay & Hhs & _ & Hcons & Heq).
    pose proof (Hmd a Hmay Hhs) as Hau. cbn [describes] in Hau.
    rewrite (field_used_any_usage _ _ _ (subkinds_closed is_local F T Hleast)
               a _ from Hcons Hau (boxed_not_local is_local Hlocal bn))
      in Heq.
    discriminate Heq.
  - destruct Hv as (closure' & args' & Heq). discriminate Heq.
Qed.

Lemma field_rebuilt_R (a : node) (i : nat) (fields fields' : list value) :
  fields_rebuilt (field_rebuilt G sol (rebuilt_value G sol) a) i fields
    fields' ->
  fields_rebuilt (field_R F T (rebuilt_value G sol) a) i fields fields'.
Proof.
  revert i fields'.
  induction fields as [ | field fields IH]; intros i [ | field' fields'] Hf;
    cbn [fields_rebuilt] in Hf |- *; try contradiction; [ exact I | ].
  destruct Hf as [Hfield Hrest]. split; [ | exact (IH (S i) fields' Hrest) ].
  intros Hk. unfold field_rebuilt in Hfield. rewrite Hk in Hfield.
  exact Hfield.
Qed.

End Subkinds_rebuilt.

Theorem rewrite_kind_with_subkind_sound :
  rewrite_kind_with_subkind_sound_statement.
Proof.
  intros is_local F T sets statics statics' x [nb sk] v v' Hleast Hlocal
    Halloc Hst Hv Hat.
  apply (subkind_sound_at is_local F T sets statics statics' Hleast Hlocal
           (rebuilt_value (graph_of_facts F) (solution_of_tables T)))
    with (v := v);
    [ exact (rebuilt_value_kind (graph_of_facts F) (solution_of_tables T))
    | exact (rebuilt_null F T)
    | exact (rebuilt_const F T)
    | exact (any_usage_not_poison is_local F T Hleast Hlocal)
    | intros s n v'' Hs; exact Hs
    | | | | | | exact Hv | exact Hat ].
  - intros tag fields n v'' (a & fields' & Hmay & Hhs & -> & Hfr).
    exists a, fields'.
    split; [ exact Hmay | split; [ exact Hhs | split; [ reflexivity | ] ] ].
    exact (field_rebuilt_R F T a 0 fields fields' Hfr).
  - intros c n v'' Hb. exact Hb.
  - intros s allocated Hl. exact (proj1 (Halloc s allocated Hl)).
  - intros s c Hl Hhu. exact (Hst s _ Hl Hhu).
  - intros s tag fields Hl Hhu.
    destruct (Hst s _ Hl Hhu) as (fields' & Hfound & Hfr).
    exists fields'. split; [ exact Hfound | ].
    exact (field_rebuilt_R F T _ 0 fields fields' Hfr).
Qed.

(* ======================================================================== *)
(* The relation of the simulation of the rebuild                            *)
(* ======================================================================== *)

(* The relation of the simulation (RP.rel) and its static environments
   (RP.statics_rel) have of the values what the proof reads, so that the
   rebuild with types, which rewrites what the parameters kept state
   ([subkinds_of_context]), states only what holds of the values that
   the rebuild gives them (RP.types_sound). *)
Section Subkinds_rel.

Variable is_local : field -> bool.
Variable F : facts.
Variable T : tables.
Variable sets : list set_of_closures.

Hypothesis Hleast : least_solution is_local F T.
Hypothesis Hlocal : is_local_slots_only is_local.
Hypothesis Hexact :
  field_used_exact is_local (graph_of_facts F) (solution_of_tables T).

Local Abbreviation G := (graph_of_facts F).
Local Abbreviation sol := (solution_of_tables T).
Local Abbreviation ctx := (plain_context is_local F T sets).

Variable kept : value_slot -> bool.
Variable called : node -> code_id -> Prop.
Variables exact partials : Prop.
Variable located : location -> option node.
Variable β : location -> location -> Prop.
Variable sets_exact : Prop.

Local Abbreviation rel :=
  (RP.rel G sol kept called exact partials located β sets_exact).

Lemma rel_null (v : value) (n : node) (v' : value) :
  rel v n v' -> v' = Value_const Null -> v = Value_const Null.
Proof.
  intros Hv ->.
  destruct v; cbn [RP.rel] in Hv; try contradiction;
    repeat match goal with
           | H : ∃ _, _ |- _ => destruct H
           | H : _ ∧ _ |- _ => destruct H
           end;
    first
      [ congruence
      | match goal with
        | H : _ = (if ?b then _ else _) |- _ => destruct b; congruence
        end ].
Qed.

Lemma rel_poison (v : value) (n : node) (v' : value) (k : flambda_kind) :
  metadata_at sol Many_sources_any_usage n -> rel v n v' ->
  v' = Value_const (Poison k) -> v = Value_const (Poison k).
Proof.
  intros Hmd Hv ->.
  destruct v; cbn [RP.rel] in Hv; try contradiction.
  all: try solve
    [ repeat match goal with
             | H : ∃ _, _ |- _ => destruct H
             | H : _ ∧ _ |- _ => destruct H
             end;
      congruence ].
  destruct Hv as (a & bn & from & Hmay & Hhs & _ & Hcons & Heq).
  pose proof (Hmd a Hmay Hhs) as Hau. cbn [describes] in Hau.
  rewrite (field_used_any_usage _ _ _ (subkinds_closed is_local F T Hleast)
             a _ from Hcons Hau (boxed_not_local is_local Hlocal bn))
    in Heq.
  discriminate Heq.
Qed.

(* The fields of a block related by the simulation, a field used having a
   use where it is held. *)
Lemma fields_rel_R (a : node) (i : nat) (fields fields' : list value) :
  RP.fields_related
    (fun i field field' =>
       ∃ from : node,
         g_constructor G a (Field_block i Value) from
         ∧ (s_field_used sol a (Field_block i Value) = true ->
            rel field from field')
         ∧ (s_field_used sol a (Field_block i Value) = false ->
            field' = Value_const (Poison Value)))
    i fields fields' ->
  fields_rebuilt (field_R F T rel a) i fields fields'.
Proof.
  revert i fields'.
  induction fields as [ | field fields IH]; intros i [ | field' fields'] Hf;
    cbn [RP.fields_related fields_rebuilt] in Hf |- *; try contradiction;
    [ exact I | ].
  destruct Hf as [(from & Hcons & Hused & Hunused) Hrest].
  split; [ | exact (IH (S i) fields' Hrest) ].
  intros _. exists from.
  split; [ exact Hcons | split; [ | exact Hunused ] ].
  intros Hu. split; [ | exact (Hused Hu) ].
  exact (used_field_hu is_local G sol (subkinds_closed is_local F T Hleast)
           Hexact _ _ _ Hu Hcons).
Qed.

(* What a parameter of the rebuild with types states holds of the value
   that the rebuild gives it, related to one of the first run of what the
   parameter stated; nothing is stated of a parameter with no use. *)
Lemma subkinds_of_context_sound (statics statics' : static_env)
    (nd : node) (nb : bool) (sk : subkind) (v v' : value) :
  RP.statics_sourced sol statics ->
  (∀ (s : symbol) (c : const),
     lookup s (static_values statics)
     = Some (Static_boxed_number (Value_const c)) ->
     s_has_usage sol (Node_symbol s) = true ->
     lookup s (static_values statics')
     = Some (Static_boxed_number (Value_const c))) ->
  (∀ (s : symbol) (tag : Z) (fields : list value),
     lookup s (static_values statics) = Some (Static_block tag fields) ->
     s_has_usage sol (Node_symbol s) = true ->
     ∃ fields' : list value,
       lookup s (static_values statics') = Some (Static_block tag fields')
       ∧ fields_rebuilt (field_R F T rel (Node_symbol s)) 0 fields fields') ->
  (s_has_usage sol nd = true -> rel v nd v') ->
  value_at statics (nb, sk) v = true ->
  value_at statics' (rewrite_kind_with_subkind ctx nd (nb, sk)) v' = true.
Proof.
  intros Hsrc Hboxed Hblocks Hrel Hat.
  destruct (s_has_usage sol nd) eqn:Hhu.
  - apply (subkind_sound_at is_local F T sets statics statics' Hleast Hlocal
             rel) with (v := v);
      [ intros v0 n v0' H; eapply RP.rel_kind; exact H
      | exact rel_null
      | intros c n v0' [_ ->]; reflexivity
      | exact rel_poison
      | intros s n v0' H; exact H
      | | intros c n v0' H; exact H
      | intros s allocated Hl; exact (proj1 (Hsrc s allocated Hl))
      | exact Hboxed | exact Hblocks | exact (Hrel eq_refl) | exact Hat ].
    intros tag fields n v0' (a & fields' & Hmay & -> & Hf & Hhs & _).
    exists a, fields'.
    split; [ exact Hmay | split; [ exact Hhs | split; [ reflexivity | ] ] ].
    exact (fields_rel_R a 0 fields fields' Hf).
  - unfold rewrite_kind_with_subkind. cbn [fst snd].
    rewrite (rewrite_subkind_erasing is_local F T sets); [ | ].
    + apply FlambdaSubkindProofs.value_at_any_value.
    + unfold metadata_for_name, get_usages.
      change (has_use ctx nd) with (s_has_usage sol nd).
      rewrite Hhu.
      destruct (get_single_source ctx nd); reflexivity.
Qed.

End Subkinds_rel.

(* The rebuild with types states only what holds, whatever the arguments
   of the relation of the simulation. *)
Lemma subkinds_of_context_sound_all (is_local : field -> bool) (F : facts)
    (T : tables) (sets : list set_of_closures)
    (code_deps : finite_map FlambdaReaperTraversal.code_dep) (e : expr) :
  least_solution is_local F T -> is_local_slots_only is_local ->
  field_used_exact is_local (graph_of_facts F) (solution_of_tables T) ->
  RP.types_sound_all (graph_of_facts F) (solution_of_tables T) F
    (subkinds_of_context (plain_context is_local F T sets) code_deps e).
Proof.
  intros Hleast Hlocal Hexact cds keep hp b ids ff ex se cc cdeps cmds
    located.
  unfold RP.types_sound, subkinds_of_context. cbv beta iota.
  cbn [FlambdaReaperRebuild.st_rewrite]. unfold rewrite_of_context.
  intros n [nb sk] v v' statics statics' Hst Hsrc Hrel Hat.
  destruct Hst as (_ & Hblocks & Honly & _).
  eapply (subkinds_of_context_sound is_local F T sets Hleast Hlocal Hexact);
    [ exact Hsrc | | | exact Hrel | exact Hat ].
  - intros s c Hl Hhu.
    destruct (Honly s _ Hl)
      as [(tag & fields & Heq) | [(slot & syms & decls & vs & Heq & _) | Ho]];
      try discriminate Heq.
    destruct Ho as [[Hn | (v0 & Hls' & Hv0)] Hbound];
      [ destruct (Hbound Hhu Hn) | ].
    rewrite Hls'. destruct Hv0 as [-> | Hsim]; [ reflexivity | ].
    cbn [RP.sim] in Hsim. rewrite Hsim. reflexivity.
  - intros s tag fields Hl Hhu.
    destruct (Hblocks s tag fields Hl Hhu) as (fields' & Hl' & Hf).
    exists fields'. split; [ exact Hl' | ].
    exact (fields_rel_R is_local F T Hleast Hexact _ _ _ _ _ _ _ _ 0 fields
             fields' Hf).
Qed.

(* The code deps that the rebuild with types reads are those given. *)
Lemma subkinds_of_context_deps (ctx : rewrite_context)
    (code_deps : finite_map FlambdaReaperTraversal.code_dep) (e : expr) :
  ∀ t : FlambdaReaperRebuild.subkinds_types,
    subkinds_of_context ctx code_deps e = Some t ->
    FlambdaReaperRebuild.st_code_deps t = code_deps.
Proof. intros t Ht. injection Ht as <-. reflexivity. Qed.

(* The table of the continuations that the rebuild with types reads is
   that of the unit. *)
Lemma subkinds_of_context_conts (ctx : rewrite_context)
    (code_deps : finite_map FlambdaReaperTraversal.code_dep) (e : expr) :
  distinct (FlambdaStaticsInScopeSpec.bound_continuations e) = true ->
  RP.conts_table_ok (subkinds_of_context ctx code_deps e) e.
Proof. intros Hd t Ht. injection Ht as <-. split; [ reflexivity | exact Hd ].
Qed.

Theorem rebuild_with_types_refines_local :
  rebuild_with_types_refines_local_statement.
Proof.
  intros flag is_local u e T sets b Hleast Hlocal Hfrag Hcf Hwf Hdist
    Hexits Hcheck Hlrs Hcheck_exact.
  exact (RP.rebuild_with_refines_local flag is_local u e T b _ Hlocal Hfrag
           Hcf Hwf Hexits Hcheck Hlrs Hcheck_exact
           (subkinds_of_context_sound_all is_local _ T sets _ _ Hleast Hlocal
              (check_field_used_exact_sound is_local _ _ Hcheck_exact))
           (subkinds_of_context_deps _ _ _)
           (subkinds_of_context_conts _ _ _ Hdist)).
Qed.

Theorem rebuild_with_types_refines_up_to_heap_local :
  rebuild_with_types_refines_up_to_heap_local_statement.
Proof.
  intros flag is_local u e T sets b Hleast Hlocal Hfrag Hdc Hrk Hwf Hdist
    Hexits Hcheck Hlrs Hcheck_exact.
  exact (RP.rebuild_with_refines_up_to_heap_local flag is_local u e T b _
           Hlocal Hfrag Hdc Hrk Hwf Hexits Hcheck Hlrs Hcheck_exact
           (subkinds_of_context_sound_all is_local _ T sets _ _ Hleast Hlocal
              (check_field_used_exact_sound is_local _ _ Hcheck_exact))
           (subkinds_of_context_deps _ _ _)
           (subkinds_of_context_conts _ _ _ Hdist)).
Qed.

From OxCamlFlambda2 Require FlambdaReaperCallingConventionsProofs.
Module CCP := FlambdaReaperCallingConventionsProofs.

Theorem rebuild_cc_with_types_refines_up_to_heap_local :
  rebuild_cc_with_types_refines_up_to_heap_local_statement.
Proof.
  intros flag is_local u e T sets cannot_change b Hleast Hlocal Hfrag Hdc Hrk
    Hwf Hdist Hexits Hcheck Hlrs Hcheck_exact Hccc Hwid.
  pose proof (CCP.traverse_witness_code_ids flag u e) as Hcid.
  assert (Hkinded : RP.calls_kinded [] e = true).
  { apply (proj1 RP.calls_kinded_of_results e [] [] [] Hrk).
    intros k l Hl. discriminate Hl. }
  destruct (CCP.check_cc_closed_sound _ _ _ Hccc) as (ccw & Hcl).
  exact (RP.exits_rel_heap _ false _ _
           (RP.rebuild_cc_with_refines_checked_local true false flag is_local
              u e T (fun n => mem_node n cannot_change) ccw b _ Hlocal Hfrag
              (proj1 (RP.any_kind_arguments_checked
                        (FlambdaReaperFlowSpec.unit_codes e)) e Hdc)
              Hkinded Hwf Hexits Hcheck Hlrs Hcheck_exact
              (subkinds_of_context_sound_all is_local _ T sets _ _ Hleast
                 Hlocal
                 (check_field_used_exact_sound is_local _ _ Hcheck_exact))
              (subkinds_of_context_deps _ _ _)
              (subkinds_of_context_conts _ _ _ Hdist) Hrk Hcl
              (CCP.check_apply_widgets_sound _ _ Hwid)
              (CCP.check_witness_code_ids_sound _ Hcid))).
Qed.

(* ======================================================================== *)
(* The reaper with types and the local unboxing                             *)
(* ======================================================================== *)

From OxCamlFlambda2 Require FlambdaReaperUnboxingRefinesHeapProofs.
Module URHP := FlambdaReaperUnboxingRefinesHeapProofs.

(* The rebuild with unboxing with types refines the rebuild with types
   without unboxing for any rewriting
   (FlambdaReaperUnboxingRefinesProofs.rebuild_unbox_with_refines_rebuild):
   the two state the same of a parameter kept and not unboxed. *)
Theorem rebuild_unbox_with_types_refines_rebuild :
  rebuild_unbox_with_types_refines_rebuild_statement.
Proof.
  intros flag is_local u e T cannot_change code_deps dominated to_unbox b
    Hslots Hfragment Hclosed Hlocal Hcheck.
  exact (URP.rebuild_unbox_with_refines_rebuild
           (subkinds_of_context
              (plain_context is_local (traverse_unit flag u e) T [])
              code_deps e)
           flag is_local u e T cannot_change code_deps dominated to_unbox b
           Hslots Hfragment Hclosed Hlocal Hcheck).
Qed.

Theorem rebuild_unbox_with_types_refines_rebuild_heap :
  rebuild_unbox_with_types_refines_rebuild_heap_statement.
Proof.
  intros flag is_local u e T cannot_change code_deps dominated to_unbox b
    Hslots Hfragment Hclosed Hlocal Hcheck.
  exact (URHP.rebuild_unbox_with_refines_rebuild_heap
           (subkinds_of_context
              (plain_context is_local (traverse_unit flag u e) T [])
              code_deps e)
           flag is_local u e T cannot_change code_deps dominated to_unbox b
           Hslots Hfragment Hclosed Hlocal Hcheck).
Qed.

(* The rebuild with types refines the unit up to observation, as the
   rebuild without types does
   (FlambdaReaperRebuildProofs.rebuild_refines_observably_local). *)
Lemma rebuild_with_types_refines_observably_local
    (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (sets : list set_of_closures) (b : bool) :
  least_solution is_local (traverse_unit flag u e) T ->
  is_local_slots_only is_local ->
  in_flow_fragment e = true ->
  FlambdaReaperRebuildSpec.any_kind_arguments e = true ->
  FlambdaReaperRebuildSpec.results_kinded [] [] e = true ->
  well_formed_unit u e = true ->
  distinct (FlambdaStaticsInScopeSpec.bound_continuations e) = true ->
  FlambdaReaperRebuildSpec.exits_one_value u e = true ->
  check_closed is_local (traverse_unit flag u e) T = true ->
  check_local_reads_used is_local (traverse_unit flag u e) T = true ->
  check_field_used_exact is_local (traverse_unit flag u e) T = true ->
  FlambdaReaperRebuildSpec.program_refines_observably e
    (rebuild_unit_with_types_from_tables is_local (traverse_unit flag u e) T
       sets (FlambdaReaperCallingConventions.unit_code_deps flag u e) b e).
Proof.
  intros Hleast Hlocal Hfrag Hdc Hrk Hwf Hdist Hexits Hcheck Hlrs
    Hcheck_exact.
  assert (Hkinded : RP.calls_kinded [] e = true).
  { apply (proj1 RP.calls_kinded_of_results e [] [] [] Hrk).
    intros k l Hl. discriminate Hl. }
  exact (proj1 (RP.exits_rel_flow false _ _
                  (RP.rebuild_with_refines_checked_local false false flag
                     is_local u e T b _ Hlocal Hfrag
                     (proj1 (RP.any_kind_arguments_checked
                               (FlambdaReaperFlowSpec.unit_codes e)) e Hdc)
                     Hkinded Hwf Hexits Hcheck Hlrs Hcheck_exact
                     (subkinds_of_context_sound_all is_local _ T sets _ _
                        Hleast Hlocal
                        (check_field_used_exact_sound is_local _ _
                           Hcheck_exact))
                     (subkinds_of_context_deps _ _ _)
                     (subkinds_of_context_conts _ _ _ Hdist)
                     (fun _ => Hrk)))).
Qed.

Theorem rebuild_unbox_with_types_refines_observably_local :
  rebuild_unbox_with_types_refines_observably_local_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox b Hleast
    Hlocal_slots Hfragment Hstatic Hkinded Hwell_formed Hdist Hone Hclosed
    Hlrs Hexact Hlocal Hcheck.
  apply (URP.refines_observably_refines _
           (State (rebuild_unit_with_types_from_tables is_local
                     (traverse_unit flag u e) T []
                     (FlambdaReaperCallingConventions.unit_code_deps flag u
                        e)
                     b e)
              [] [] empty_static_env empty_heap no_reads)).
  - exact (rebuild_with_types_refines_observably_local flag is_local u e T []
             b Hleast Hlocal_slots Hfragment
             (RP.static_constants_any_kind_arguments _ e Hstatic)
             Hkinded Hwell_formed Hdist Hone Hclosed Hlrs Hexact).
  - exact (rebuild_unbox_with_types_refines_rebuild flag is_local u e T
             cannot_change
             (FlambdaReaperCallingConventions.unit_code_deps flag u e)
             dominated to_unbox b Hlocal_slots Hfragment Hclosed Hlocal
             Hcheck).
Qed.

Theorem rebuild_unbox_with_types_refines_observably :
  rebuild_unbox_with_types_refines_observably_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox b Hleast
    Hlocal_slots Hprojections Hfragment Hstatic Hkinded Hwell_formed Hdist
    Hone Hclosed Hexact Hlocal Hcheck.
  exact (rebuild_unbox_with_types_refines_observably_local flag is_local u e
           T cannot_change dominated to_unbox b Hleast Hlocal_slots
           Hfragment Hstatic Hkinded Hwell_formed Hdist Hone Hclosed
           (FlambdaReaperFlowProofs.old_premise_local_reads_used flag
              is_local u e T Hlocal_slots Hprojections)
           Hexact Hlocal Hcheck).
Qed.

Theorem rebuild_unbox_with_types_refines_up_to_heap_local :
  rebuild_unbox_with_types_refines_up_to_heap_local_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox b Hleast
    Hlocal_slots Hfragment Hstatic Hkinded Hwell_formed Hdist Hone Hclosed
    Hlrs Hexact Hlocal Hcheck.
  apply (URHP.refines_up_to_heap_refines _
           (State (rebuild_unit_with_types_from_tables is_local
                     (traverse_unit flag u e) T []
                     (FlambdaReaperCallingConventions.unit_code_deps flag u
                        e)
                     b e)
              [] [] empty_static_env empty_heap no_reads)).
  - exact (rebuild_with_types_refines_up_to_heap_local flag is_local u e T []
             b Hleast Hlocal_slots Hfragment
             (RP.static_constants_any_kind_arguments _ e Hstatic)
             Hkinded Hwell_formed Hdist Hone Hclosed Hlrs Hexact).
  - exact (rebuild_unbox_with_types_refines_rebuild_heap flag is_local u e T
             cannot_change
             (FlambdaReaperCallingConventions.unit_code_deps flag u e)
             dominated to_unbox b Hlocal_slots Hfragment Hclosed Hlocal
             Hcheck).
Qed.

Theorem rebuild_unbox_with_types_refines_up_to_heap :
  rebuild_unbox_with_types_refines_up_to_heap_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox b Hleast
    Hlocal_slots Hprojections Hfragment Hstatic Hkinded Hwell_formed Hdist
    Hone Hclosed Hexact Hlocal Hcheck.
  exact (rebuild_unbox_with_types_refines_up_to_heap_local flag is_local u e
           T cannot_change dominated to_unbox b Hleast Hlocal_slots
           Hfragment Hstatic Hkinded Hwell_formed Hdist Hone Hclosed
           (FlambdaReaperFlowProofs.old_premise_local_reads_used flag
              is_local u e T Hlocal_slots Hprojections)
           Hexact Hlocal Hcheck).
Qed.

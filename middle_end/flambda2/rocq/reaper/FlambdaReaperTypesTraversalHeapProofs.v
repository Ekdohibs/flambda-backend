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

(* The proofs of FlambdaReaperTypesTraversalHeapSpec.v. Nothing here
   needs review.

   The graph of the traversal: the proofs of
   FlambdaReaperTypesTraversalProofs.v and
   FlambdaReaperTypesTraversalUnaliasedProofs.v read of the fragment only
   that the unit has no older form of set of closures, which
   [in_heap_fragment] gives as [in_flow_fragment] does
   ([TT.unit_premises_heap]); they are stated there for
   [in_heap_fragment] ([TT.traversal_props_heap],
   [TU.closures_unaliased_traversal_heap]).

   The statements of a graph and its tables follow from those without
   the heap by erasing the locations of the three runs
   (FlambdaReaperTypesHeapSpec.erase_locations), as in
   FlambdaReaperTypesHeapProofs.v: the erasure keeps what types say
   ([THP.type_satisfied_erase]), makes of [rebuilt_value_heap]
   [rebuilt_value] ([THP.rebuilt_value_heap_erase]), and keeps the values
   at the paths of fields ([value_at_path_erase]), so [unboxed_values]
   and [changed_at] ([unboxed_values_erase], [changed_at_erase]). The
   statements of the traversal of a unit follow as without the heap. So
   do those without [changed_unstored]: the erasure makes of
   [rebuilt_changed_heap] [rebuilt_changed] ([rebuilt_changed_heap_erase],
   the erased pairs being closed under its step, [erased_closed]).

   The exported types of the unit rebuilt with local unboxing: those of
   the plain rebuild, with the heap
   (FlambdaReaperTypesHeapProofs.types_end_to_end_term_heap_gen and its
   forms), hold of the static environment where the unit rebuilt with
   unboxing exits, which is that of the plain rebuild
   ([THX.unbox_exit_heap], of the simulation of
   FlambdaReaperUnboxingRefinesHeapProofs.v). *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaValueSpec FlambdaPrimitiveSpec FlambdaSemanticsSpec
  FlambdaWellFormedSpec FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperSolution FlambdaReaperRebuild FlambdaReaperFlowSpec
  FlambdaReaperLeastSpec FlambdaReaperLocalFields FlambdaReaperTypes
  FlambdaReaperTypesSpec FlambdaReaperTypesHeapSpec
  FlambdaReaperTypesTraversalSpec FlambdaReaperTypesTraversalUnaliasedSpec
  FlambdaReaperTypesTraversalHeapSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypingEnv FlambdaTypingEnvSpec.
From OxCamlFlambda2 Require FlambdaReaperUnboxing
  FlambdaReaperUnboxingRefinesCallsSpec FlambdaReaperTypesProofs
  FlambdaReaperTypesHeapProofs FlambdaReaperTypesTraversalProofs
  FlambdaReaperTypesTraversalUnaliasedProofs FlambdaReaperLeastProofs
  FlambdaReaperTypesTraversalHeapExitProofs.
Import ListNotations.

Module TP := FlambdaReaperTypesProofs.
Module THP := FlambdaReaperTypesHeapProofs.
Module TT := FlambdaReaperTypesTraversalProofs.
Module TU := FlambdaReaperTypesTraversalUnaliasedProofs.
Module THX := FlambdaReaperTypesTraversalHeapExitProofs.

Local Abbreviation er := erase_locations.

(* ======================================================================== *)
(* The graph of the traversal                                               *)
(* ======================================================================== *)

Theorem function_slots_mutual_traversal_heap :
  function_slots_mutual_traversal_heap_statement.
Proof.
  intros u e Hfrag Hwfu Honce.
  exact (proj1 (TT.traversal_props_heap u e Hfrag Hwfu Honce)).
Qed.

Theorem function_slots_of_sets_traversal_heap :
  function_slots_of_sets_traversal_heap_statement.
Proof.
  intros u e Hfrag Hwfu Honce.
  exact (proj2 (TT.traversal_props_heap u e Hfrag Hwfu Honce)).
Qed.

Theorem closures_unaliased_traversal_heap :
  closures_unaliased_traversal_heap_statement.
Proof. exact TU.closures_unaliased_traversal_heap. Qed.

(* ======================================================================== *)
(* The erasure of locations and the paths of fields                         *)
(* ======================================================================== *)

Lemma field_of_value_erase (f : field) (w : value) :
  field_of_value f (er w) = option_map er (field_of_value f w).
Proof.
  destruct f, w; cbn [field_of_value erase_locations]; try reflexivity.
  apply nth_error_map.
Qed.

Lemma value_at_path_erase (path : list field) :
  ∀ w : value, value_at_path path (er w) = option_map er (value_at_path path w).
Proof.
  induction path as [ | f path IH]; intros w; cbn [value_at_path];
    [ reflexivity | ].
  rewrite field_of_value_erase.
  destruct (field_of_value f w) as [w1 | ]; cbn [option_map];
    [ exact (IH w1) | reflexivity ].
Qed.

Lemma unboxed_values_erase (fields : FlambdaReaperUnboxing.unboxed_fields)
    (w : value) (ws : list value) :
  unboxed_values fields w ws -> unboxed_values fields (er w) (map er ws).
Proof.
  unfold unboxed_values.
  generalize (FlambdaReaperUnboxingRefinesCallsSpec.fields_paths fields)
    as P.
  induction 1 as [ | pk w' P ws' [Hv Hk] _ IH]; cbn [map]; constructor;
    [ split | exact IH ].
  - rewrite value_at_path_erase, Hv. reflexivity.
  - rewrite THP.erase_kind. exact Hk.
Qed.

Lemma Forall2_sat_unerase (statics : static_env) (vars : env)
    (ws : list value) :
  ∀ ts : list flambda_type,
    Forall2 (type_satisfied (erase_statics statics) (erase_env vars))
      (map er ws) ts ->
    Forall2 (type_satisfied statics vars) ws ts.
Proof.
  intros ts H. remember (map er ws) as l eqn:El. revert ws El.
  induction H as [ | x t l ts' Hx _ IH]; intros ws El.
  - destruct ws; [ constructor | discriminate El ].
  - destruct ws as [ | w ws]; cbn [map] in El; [ discriminate El | ].
    injection El as -> El. constructor.
    + exact (proj1 (THP.type_satisfied_erase statics vars w t) Hx).
    + exact (IH ws El).
Qed.

Lemma map_fst_erase_slots (vs : list (value_slot * value)) :
  map fst (erase_slots vs) = map fst vs.
Proof.
  unfold erase_slots. rewrite map_map. apply map_ext.
  intros [w x]. reflexivity.
Qed.

Lemma map_snd_erase_slots (vs : list (value_slot * value)) :
  map snd (erase_slots vs) = map er (map snd vs).
Proof.
  unfold erase_slots. rewrite !map_map. apply map_ext.
  intros [w x]. reflexivity.
Qed.

Lemma slot_value_at_path_erase (vs : list (value_slot * value))
    (path : list field) :
  slot_value_at_path (erase_slots vs) path
  = option_map er (slot_value_at_path vs path).
Proof.
  destruct path as [ | f path]; [ reflexivity | ].
  destruct f as [i k | w | g | entry | | | bn | r | ];
    cbn [slot_value_at_path]; try reflexivity.
  rewrite THP.lookup_erase_slots.
  destruct (lookup w vs) as [x | ]; cbn [option_map];
    [ exact (value_at_path_erase path x) | reflexivity ].
Qed.

Lemma changed_closure_rebuilt_erase (r : FlambdaReaperUnboxing.closure_repr)
    (v v' : value) :
  changed_closure_rebuilt r v v' -> changed_closure_rebuilt r (er v) (er v').
Proof.
  intros (decls & vs & vs' & -> & -> & Hlk & Hfst & Hall).
  exists decls, (erase_slots vs), (erase_slots vs').
  split; [ reflexivity | split; [ reflexivity | split; [ exact Hlk | ] ] ].
  split; [ rewrite map_fst_erase_slots; exact Hfst | ].
  rewrite map_snd_erase_slots. revert Hall.
  generalize (FlambdaReaperUnboxingRefinesCallsSpec.fields_paths
                (FlambdaReaperUnboxing.cr_fields r)) as P.
  generalize (map snd vs') as xs. intros xs P Hall.
  induction Hall as [ | pk x' P xs [Hp Hk] _ IH]; cbn [map]; constructor;
    [ split | exact IH ].
  - rewrite slot_value_at_path_erase, Hp. reflexivity.
  - rewrite THP.erase_kind. exact Hk.
Qed.

Lemma changed_at_erase (sol : solution)
    (reprs : list (node * FlambdaReaperUnboxing.closure_repr)) (n : node)
    (v v' : value) :
  changed_at sol reprs n v v' -> changed_at sol reprs n (er v) (er v').
Proof.
  intros [Hex Hall]. split; [ exact Hex | ].
  intros a Hmay Hhs. destruct (Hall a Hmay Hhs) as (r & Hlk & Hcr).
  exists r.
  split; [ exact Hlk | exact (changed_closure_rebuilt_erase r _ _ Hcr) ].
Qed.

Lemma closure_in_set_erase_not_none (statics : static_env) (v : value) :
  closure_in_set statics v <> None ->
  closure_in_set (erase_statics statics) (er v) <> None.
Proof.
  intros H. rewrite THP.closure_in_set_erase.
  destruct (closure_in_set statics v); cbn [option_map];
    [ discriminate | exact H ].
Qed.

(* ======================================================================== *)
(* The statements of a graph and its tables                                 *)
(* ======================================================================== *)

Lemma changed_slots_heap_gen (fx : fixes) :
  rewrite_changed_type_sound_slots_with fx ->
  rewrite_changed_type_sound_slots_heap_with fx.
Proof.
  intros Hnh is_local F T sets dominated to_unbox tcr e tenv canon statics
    statics' valuation valuation' β fuel m t n v v_mid v' Hleast Hlocal
    Hsets Hkinds Hmut Henv Hfrag Halloc Hst Hwhole Hcopies Hdomc Huses Hunst
    Hreprs Ht Hcis Hrv Hat Hch Hsat.
  apply (proj1 (THP.type_satisfied_erase statics' valuation' v' _)).
  exact (Hnh is_local F T sets dominated to_unbox tcr e tenv canon
           (erase_statics statics) (erase_statics statics')
           (erase_env valuation) (erase_env valuation') fuel m t n (er v)
           (er v_mid) (er v') Hleast Hlocal Hsets Hkinds Hmut
           (THP.typing_env_satisfied_strictly_erase _ _ _ Henv)
           (THP.env_in_fragment_erase _ _ _ Hfrag)
           (THP.statics_allocated_erase _ _ Halloc)
           (THP.statics_rebuilt_heap_erase _ _ _ _ _ Hst)
           (THP.static_sets_whole_erase _ _ _ Hwhole)
           (THP.copies_valued_erase _ _ _ _ Hcopies) Hdomc Huses Hunst Hreprs
           (THP.type_in_fragment_erase _ _ _ Ht)
           (closure_in_set_erase_not_none _ _ Hcis)
           (THP.rebuilt_value_heap_erase _ _ _ _ _ _ Hrv) Hat
           (changed_at_erase _ _ _ _ _ Hch)
           (proj2 (THP.type_satisfied_erase _ _ _ _) Hsat)).
Qed.

Theorem rewrite_changed_type_sound_slots_heap :
  rewrite_changed_type_sound_slots_heap_statement.
Proof.
  exact (changed_slots_heap_gen _ TP.rewrite_changed_type_sound_slots).
Qed.

Theorem rewrite_changed_type_sound_slots_heap_landed :
  rewrite_changed_type_sound_slots_heap_landed_statement.
Proof.
  exact (changed_slots_heap_gen _ TP.rewrite_changed_type_sound_slots_landed).
Qed.

Lemma unboxed_result_heap_gen (local loose : bool) (fx : fixes) :
  TP.fixes_sound fx -> unboxed_result_type_sound_heap_with local loose fx.
Proof.
  intros (Hb & Hf & Hr & Hn) is_local F T sets dominated to_unbox e tenv
    canon statics statics' valuation valuation' β code_var type_var fields v
    v_mid ws Hleast Hlocal Hsets Hkinds Henv Hfrag Halloc Hst Hwhole Hcopies
    Huses Hlk Hv Hrv Hws.
  assert (Hle : TP.least_either is_local F T)
    by (destruct local; [ right | left ]; exact Hleast).
  apply Forall2_sat_unerase.
  exact (TP.rewrite_unboxed_result_type_sound_in fx Hb Hf Hr Hn is_local F T
           sets tenv canon (erase_statics statics) (erase_statics statics')
           (erase_env valuation) (erase_env valuation') loose dominated
           to_unbox e Hle Hlocal Hsets Hkinds
           (THP.typing_env_satisfied_strictly_erase _ _ _ Henv)
           (THP.env_in_fragment_erase _ _ _ Hfrag)
           (THP.statics_allocated_erase _ _ Halloc)
           (THP.statics_rebuilt_heap_erase _ _ _ _ _ Hst)
           (THP.copies_valued_erase _ _ _ _ Hcopies)
           (fun H => THP.static_sets_whole_erase _ _ _ (Hwhole H)) Huses
           code_var type_var fields (er v) (er v_mid) (map er ws) Hlk
           (eq_trans (THP.lookup_erase_env valuation type_var)
              (f_equal (option_map er) Hv))
           (THP.rebuilt_value_heap_erase _ _ _ _ _ _ Hrv)
           (unboxed_values_erase _ _ _ Hws)).
Qed.

Theorem rewrite_unboxed_result_type_sound_heap :
  rewrite_unboxed_result_type_sound_heap_statement.
Proof. exact (unboxed_result_heap_gen false false _ TP.all_fixes_sound). Qed.

Theorem rewrite_unboxed_result_type_sound_heap_landed :
  rewrite_unboxed_result_type_sound_heap_landed_statement.
Proof.
  exact (unboxed_result_heap_gen false false _ TP.landed_fixes_sound).
Qed.

Theorem rewrite_unaliased_unboxed_result_type_sound_heap :
  rewrite_unaliased_unboxed_result_type_sound_heap_statement.
Proof. exact (unboxed_result_heap_gen false true _ TP.all_fixes_sound). Qed.

Theorem unaliased_unboxed_result_type_sound_heap_landed :
  unaliased_unboxed_result_type_sound_heap_landed_statement.
Proof. exact (unboxed_result_heap_gen false true _ TP.landed_fixes_sound). Qed.

Theorem rewrite_unboxed_result_type_sound_heap_local :
  rewrite_unboxed_result_type_sound_heap_local_statement.
Proof. exact (unboxed_result_heap_gen true false _ TP.all_fixes_sound). Qed.

Theorem unboxed_result_type_sound_heap_local_landed :
  unboxed_result_type_sound_heap_local_landed_statement.
Proof. exact (unboxed_result_heap_gen true false _ TP.landed_fixes_sound). Qed.

Theorem unaliased_unboxed_result_type_sound_heap_local :
  unaliased_unboxed_result_type_sound_heap_local_statement.
Proof. exact (unboxed_result_heap_gen true true _ TP.all_fixes_sound). Qed.

Theorem unaliased_unboxed_result_sound_heap_local_landed :
  unaliased_unboxed_result_sound_heap_local_landed_statement.
Proof. exact (unboxed_result_heap_gen true true _ TP.landed_fixes_sound). Qed.

(* ======================================================================== *)
(* The statements of the traversal of a unit                                *)
(* ======================================================================== *)

Lemma changed_traversal_heap_gen (fx : fixes) :
  rewrite_changed_type_sound_slots_with fx ->
  rewrite_changed_type_sound_traversal_heap_with fx.
Proof.
  intros Hnh is_local u T sets dominated to_unbox tcr e tenv canon statics
    statics' valuation valuation' β fuel m t n v v_mid v' Hfrag Hwfu Honce
    Hleast Hlocal Hsets Hkinds Henv Hfragt Halloc Hst Hwhole Hcopies Hdomc
    Huses Hunst Hone Ht Hcis Hrv Hat Hch Hsat.
  destruct (TT.traversal_props_heap u e Hfrag Hwfu Honce) as [Hmut Hfs].
  exact (changed_slots_heap_gen fx Hnh is_local (traverse_unit Auto u e) T
           sets dominated to_unbox tcr e tenv canon statics statics'
           valuation valuation' β fuel m t n v v_mid v' Hleast Hlocal Hsets
           Hkinds Hmut Henv Hfragt Halloc Hst Hwhole Hcopies Hdomc Huses
           Hunst
           (TT.reprs_of_sets_derived is_local _ T dominated to_unbox tcr e
              Hleast Hfs Hmut Hdomc Hone)
           Ht Hcis Hrv Hat Hch Hsat).
Qed.

Lemma changed_unaliased_heap_gen (fx : fixes) :
  rewrite_changed_type_sound_slots_with fx ->
  changed_type_sound_traversal_heap_with fx.
Proof.
  intros Hnh is_local u T sets dominated to_unbox tcr e tenv canon statics
    statics' valuation valuation' β fuel m t n v v_mid v' Hfrag Hwfu Honce
    Hleast Hlocal Hsets Henv Hfragt Halloc Hst Hwhole Hcopies Hdomc Huses
    Hunst Hu Ht Hcis Hrv Hat Hch Hsat.
  destruct (TT.traversal_props_heap u e Hfrag Hwfu Honce) as [Hmut Hfs].
  exact (changed_traversal_heap_gen fx Hnh is_local u T sets dominated
           to_unbox tcr e tenv canon statics statics' valuation valuation' β
           fuel m t n v v_mid v' Hfrag Hwfu Honce Hleast Hlocal Hsets
           (TT.block_fields_of_one_kind_traversal Auto u e) Henv Hfragt
           Halloc Hst Hwhole Hcopies Hdomc Huses Hunst
           (TT.changed_closures_one_allocation_unaliased is_local _ T tcr
              Hleast Hfs Hmut Hu)
           Ht Hcis Hrv Hat Hch Hsat).
Qed.

Lemma changed_full_heap_gen (fx : fixes) :
  rewrite_changed_type_sound_slots_with fx ->
  changed_type_sound_full_heap_with fx.
Proof.
  intros Hnh is_local u T sets dominated to_unbox tcr e tenv canon statics
    statics' valuation valuation' β fuel m t n v v_mid v' Hfrag Hwfu Honce
    Hleast Hlocal Hsets Henv Hfragt Halloc Hst Hwhole Hcopies Hdomc Huses
    Hunst Ht Hcis Hrv Hat Hch Hsat.
  exact (changed_unaliased_heap_gen fx Hnh is_local u T sets dominated
           to_unbox tcr e tenv canon statics statics' valuation valuation' β
           fuel m t n v v_mid v' Hfrag Hwfu Honce Hleast Hlocal Hsets Henv
           Hfragt Halloc Hst Hwhole Hcopies Hdomc Huses Hunst
           (TU.closures_unaliased_traversal_heap u e Hfrag Hwfu Honce)
           Ht Hcis Hrv Hat Hch Hsat).
Qed.

Theorem rewrite_changed_type_sound_traversal_heap :
  rewrite_changed_type_sound_traversal_heap_statement.
Proof.
  exact (changed_traversal_heap_gen _ TP.rewrite_changed_type_sound_slots).
Qed.

Theorem rewrite_changed_type_sound_traversal_heap_landed :
  rewrite_changed_type_sound_traversal_heap_landed_statement.
Proof.
  exact (changed_traversal_heap_gen _
           TP.rewrite_changed_type_sound_slots_landed).
Qed.

Theorem changed_type_sound_traversal_heap :
  changed_type_sound_traversal_heap_statement.
Proof.
  exact (changed_unaliased_heap_gen _ TP.rewrite_changed_type_sound_slots).
Qed.

Theorem changed_type_sound_traversal_heap_landed :
  changed_type_sound_traversal_heap_landed_statement.
Proof.
  exact (changed_unaliased_heap_gen _
           TP.rewrite_changed_type_sound_slots_landed).
Qed.

Theorem changed_type_sound_full_heap : changed_type_sound_full_heap_statement.
Proof.
  exact (changed_full_heap_gen _ TP.rewrite_changed_type_sound_slots).
Qed.

Theorem changed_type_sound_full_heap_landed :
  changed_type_sound_full_heap_landed_statement.
Proof.
  exact (changed_full_heap_gen _ TP.rewrite_changed_type_sound_slots_landed).
Qed.

(* ======================================================================== *)
(* Closures that change, wherever they are held                            *)
(* ======================================================================== *)

Section Changed_erase.

Variable G : graph.
Variable sol : solution.
Variable reprs : list (node * FlambdaReaperUnboxing.closure_repr).
Variable β : location -> location -> Prop.

Local Abbreviation RCH := (rebuilt_changed_heap G sol reprs β).

Lemma rch_unfold (y : value) (k : node) (y' : value) :
  RCH y k y' -> rebuilt_changed_heap_step G sol reprs β RCH y k y'.
Proof.
  intros (R & HR & Hy). destruct (HR _ _ _ Hy) as [H | H]; [ left | right ].
  - apply (TP.step_impl G sol reprs R); [ | exact H ].
    intros x m x' Hx. exists R. split; [ exact HR | exact Hx ].
  - exact H.
Qed.

(* The values of the first and the third runs, erased. *)
Definition erased_rel (x : value) (k : node) (x' : value) : Prop :=
  ∃ y y' : value, x = er y ∧ x' = er y' ∧ RCH y k y'.

Lemma slots_changed_erase (decls : list (function_slot * code_id))
    (sites : function_slot -> node) (vs vs' : list (value_slot * value)) :
  for_all_bindings (value_slot_rebuilt G sol RCH decls sites vs') vs ->
  for_all_bindings
    (value_slot_rebuilt G sol erased_rel decls sites (erase_slots vs'))
    (erase_slots vs).
Proof.
  intros H. apply (proj2 (FlambdaMapProofs.for_all_bindings_lookup _ _ _)).
  intros w x0 Hl. rewrite THP.lookup_erase_slots in Hl.
  destruct (lookup w vs) as [x | ] eqn:E; cbn [option_map] in Hl;
    [ | discriminate Hl ].
  injection Hl as <-.
  pose proof (proj1 (FlambdaMapProofs.for_all_bindings_lookup _ _ _) H w x E)
    as Hx.
  intros f id Hf Hused. destruct (Hx f id Hf Hused)
    as (from & x' & Hheld & Hhs & Hhu & Hl' & Hrv).
  exists from, (er x').
  split; [ exact Hheld | split; [ exact Hhs | split; [ exact Hhu | ] ] ].
  split; [ rewrite THP.lookup_erase_slots, Hl'; reflexivity | ].
  exists x, x'. split; [ reflexivity | split; [ reflexivity | exact Hrv ] ].
Qed.

Lemma fields_changed_erase (a : node) (fields fields' : list value) :
  fields_rebuilt (field_rebuilt G sol RCH a) 0 fields fields' ->
  fields_rebuilt (field_rebuilt G sol erased_rel a) 0 (map er fields)
    (map er fields').
Proof.
  intros Hf. refine (THP.fields_rebuilt_map _ _ er fields _ 0 fields' Hf).
  apply Forall_forall. intros x _ i x' (from & Hc & Hu & Hn).
  unfold field_rebuilt. rewrite THP.erase_kind. exists from.
  split; [ exact Hc | split ].
  - intros Hused. destruct (Hu Hused) as [Hhu Hrv]. split; [ exact Hhu | ].
    exists x, x'. split; [ reflexivity | split; [ reflexivity | exact Hrv ] ].
  - intros Hunused. rewrite (Hn Hunused). reflexivity.
Qed.

Lemma erased_closed : rebuilt_changed_closed G sol reprs erased_rel.
Proof.
  intros x k x' (y & y' & -> & -> & Hy).
  apply rch_unfold in Hy. destruct Hy as [Hy | (l & l' & -> & -> & _)].
  2:{ cbn [erase_locations rebuilt_changed_step].
      exists [Value_location l']. reflexivity. }
  destruct y as [c | code vs0 | slot decls vs | s | tag fields | elements
                | c | l | callee args arity];
    cbn [rebuilt_changed_step] in Hy;
    cbn [erase_locations rebuilt_changed_step]; try contradiction.
  - destruct Hy as [Has ->]. split; [ exact Has | reflexivity ].
  - destruct Hy as (sites & vs' & Hmay & Hsites & Hall & Hm).
    exists sites, (erase_slots vs').
    split; [ exact Hmay | split; [ exact Hsites | split ] ].
    + exact (slots_changed_erase _ _ _ _ Hall).
    + destruct (allocation_repr sol reprs (sites slot)) as [r | ].
      * exact (changed_closure_rebuilt_erase r _ _ Hm).
      * subst y'. reflexivity.
  - destruct Hy as (-> & H1 & H2 & H3). split; [ reflexivity | auto ].
  - destruct Hy as (a & fields' & Hmay & Hhs & -> & Hf).
    exists a, (map er fields').
    split; [ exact Hmay | split; [ exact Hhs | split; [ reflexivity | ] ] ].
    exact (fields_changed_erase a fields fields' Hf).
  - destruct Hy as (elements' & ->). exists (map er elements'). reflexivity.
  - destruct Hy as (a & bn & from & Hmay & Hhs & Hk & Hc & ->).
    exists a, bn, from. repeat (split; [ assumption | ]).
    destruct (s_field_used sol a (Field_boxed_number bn)); reflexivity.
  - destruct Hy as (closure' & args' & ->).
    exists (er closure'), (map er args'). reflexivity.
Qed.

Lemma erased_rc (x : value) (k : node) (x' : value) :
  erased_rel x k x' -> rebuilt_changed G sol reprs x k x'.
Proof.
  exact (TP.rebuilt_changed_of_closed G sol reprs _ x k x' erased_closed).
Qed.

(* A value of the first run and what the third has for it, with the heap,
   erased, are related without. *)
Lemma rebuilt_changed_heap_erase (v : value) (n : node) (v' : value) :
  RCH v n v' -> rebuilt_changed G sol reprs (er v) n (er v').
Proof.
  intros H. apply erased_rc. exists v, v'.
  split; [ reflexivity | split; [ reflexivity | exact H ] ].
Qed.

Lemma statics_changed_heap_erase (statics statics' : static_env) :
  statics_rebuilt_changed_heap G sol reprs β statics statics' ->
  statics_rebuilt_changed G sol reprs (erase_statics statics)
    (erase_statics statics').
Proof.
  intros H s a0 Hl Hhu Hnc. rewrite THP.lookup_erase_statics in Hl.
  destruct (lookup s (static_values statics)) as [a | ] eqn:E;
    cbn [option_map] in Hl; [ | discriminate Hl ].
  injection Hl as <-. pose proof (H s a E Hhu Hnc) as Hs.
  rewrite THP.lookup_erase_statics.
  destruct a as [code vs | slot syms decls vs | tag fields | contents | bytes
                | elements | c];
    cbn [static_rebuilt_with] in Hs; cbn [erase_static static_rebuilt_with].
  - exact Hs.
  - destruct Hs as (sites & vs' & -> & Hsite & Hsites & Hall).
    exists sites, (erase_slots vs').
    split; [ reflexivity | split; [ exact Hsite | split; [ exact Hsites | ] ] ].
    refine (TP.for_all_bindings_impl _ _ _ _
              (slots_changed_erase _ _ _ _ Hall)).
    intros w x Hw.
    exact (TP.value_slot_rebuilt_impl G sol _ _ _ _ _ w x erased_rc Hw).
  - destruct Hs as (fields' & -> & Hf). exists (map er fields').
    split; [ reflexivity | ].
    refine (TP.fields_rebuilt_impl _ _ _ _ _ _
              (fields_changed_erase _ _ _ Hf)).
    intros i x x' Hx.
    exact (TP.field_rebuilt_impl G sol _ _ _ i x x' erased_rc Hx).
  - rewrite Hs. reflexivity.
  - rewrite Hs. reflexivity.
  - destruct Hs as (elements' & ->). exists (map er elements'). reflexivity.
  - rewrite Hs. reflexivity.
Qed.

End Changed_erase.

Lemma static_sets_whole_unchanged_erase (sol : solution)
    (reprs : list (node * FlambdaReaperUnboxing.closure_repr))
    (statics statics' : static_env) :
  static_sets_whole_unchanged sol reprs statics statics' ->
  static_sets_whole_unchanged sol reprs (erase_statics statics)
    (erase_statics statics').
Proof.
  intros H s slot syms decls vs0 Hl Hhu Hnc.
  rewrite THP.lookup_erase_statics in Hl.
  destruct (lookup s (static_values statics)) as [a | ] eqn:E;
    cbn [option_map] in Hl; [ | discriminate Hl ].
  destruct a; cbn [erase_static] in Hl; try discriminate Hl.
  injection Hl as <- <- <- <-.
  destruct (H s _ _ _ _ E Hhu Hnc) as [Hslot (vs' & Hall)].
  split; [ exact Hslot | ]. exists (erase_slots vs').
  intros g s_g Hg. destruct (Hall g s_g Hg) as [H1 H2].
  rewrite !THP.lookup_erase_statics, H1, H2. split; reflexivity.
Qed.

Lemma stored_slots_heap_gen (fx : fixes) :
  rewrite_stored_type_sound_slots_with fx ->
  rewrite_stored_type_sound_slots_heap_with fx.
Proof.
  intros Hnh is_local F T sets dominated to_unbox tcr e tenv canon statics
    statics' valuation valuation' β fuel m t n v v' Hleast Hlocal Hsets
    Hkinds Hmut Henv Hfrag Halloc Hst Hwhole Hcopies Hdomc Huses Hreprs Ht
    Hrv Hat Hsat.
  apply (proj1 (THP.type_satisfied_erase statics' valuation' v' _)).
  exact (Hnh is_local F T sets dominated to_unbox tcr e tenv canon
           (erase_statics statics) (erase_statics statics')
           (erase_env valuation) (erase_env valuation') fuel m t n (er v)
           (er v') Hleast Hlocal Hsets Hkinds Hmut
           (THP.typing_env_satisfied_strictly_erase _ _ _ Henv)
           (THP.env_in_fragment_erase _ _ _ Hfrag)
           (THP.statics_allocated_erase _ _ Halloc)
           (statics_changed_heap_erase _ _ _ _ _ _ Hst)
           (static_sets_whole_unchanged_erase _ _ _ _ Hwhole)
           (THP.copies_valued_erase _ _ _ _ Hcopies) Hdomc Huses Hreprs
           (THP.type_in_fragment_erase _ _ _ Ht)
           (rebuilt_changed_heap_erase _ _ _ _ _ _ _ Hrv) Hat
           (proj2 (THP.type_satisfied_erase _ _ _ _) Hsat)).
Qed.

Theorem rewrite_stored_type_sound_slots_heap :
  rewrite_stored_type_sound_slots_heap_statement.
Proof. exact (stored_slots_heap_gen _ TP.rewrite_stored_type_sound_slots). Qed.

Theorem rewrite_stored_type_sound_slots_heap_landed :
  rewrite_stored_type_sound_slots_heap_landed_statement.
Proof.
  exact (stored_slots_heap_gen _ TP.rewrite_stored_type_sound_slots_landed).
Qed.

Lemma stored_traversal_heap_gen (fx : fixes) :
  rewrite_stored_type_sound_slots_with fx ->
  rewrite_stored_type_sound_traversal_heap_with fx.
Proof.
  intros Hnh is_local u T sets dominated to_unbox tcr e tenv canon statics
    statics' valuation valuation' β fuel m t n v v' Hfrag Hwfu Honce Hleast
    Hlocal Hsets Hkinds Henv Hfragt Halloc Hst Hwhole Hcopies Hdomc Huses
    Hone Ht Hrv Hat Hsat.
  destruct (TT.traversal_props_heap u e Hfrag Hwfu Honce) as [Hmut Hfs].
  exact (stored_slots_heap_gen fx Hnh is_local (traverse_unit Auto u e) T
           sets dominated to_unbox tcr e tenv canon statics statics'
           valuation valuation' β fuel m t n v v' Hleast Hlocal Hsets Hkinds
           Hmut Henv Hfragt Halloc Hst Hwhole Hcopies Hdomc Huses
           (TT.reprs_of_sets_derived is_local _ T dominated to_unbox tcr e
              Hleast Hfs Hmut Hdomc Hone)
           Ht Hrv Hat Hsat).
Qed.

Lemma stored_unaliased_heap_gen (fx : fixes) :
  rewrite_stored_type_sound_slots_with fx ->
  stored_type_sound_traversal_heap_with fx.
Proof.
  intros Hnh is_local u T sets dominated to_unbox tcr e tenv canon statics
    statics' valuation valuation' β fuel m t n v v' Hfrag Hwfu Honce Hleast
    Hlocal Hsets Henv Hfragt Halloc Hst Hwhole Hcopies Hdomc Huses Hu Ht Hrv
    Hat Hsat.
  destruct (TT.traversal_props_heap u e Hfrag Hwfu Honce) as [Hmut Hfs].
  exact (stored_traversal_heap_gen fx Hnh is_local u T sets dominated
           to_unbox tcr e tenv canon statics statics' valuation valuation' β
           fuel m t n v v' Hfrag Hwfu Honce Hleast Hlocal Hsets
           (TT.block_fields_of_one_kind_traversal Auto u e) Henv Hfragt
           Halloc Hst Hwhole Hcopies Hdomc Huses
           (TT.changed_closures_one_allocation_unaliased is_local _ T tcr
              Hleast Hfs Hmut Hu)
           Ht Hrv Hat Hsat).
Qed.

Lemma stored_full_heap_gen (fx : fixes) :
  rewrite_stored_type_sound_slots_with fx ->
  stored_type_sound_full_heap_with fx.
Proof.
  intros Hnh is_local u T sets dominated to_unbox tcr e tenv canon statics
    statics' valuation valuation' β fuel m t n v v' Hfrag Hwfu Honce Hleast
    Hlocal Hsets Henv Hfragt Halloc Hst Hwhole Hcopies Hdomc Huses Ht Hrv Hat
    Hsat.
  exact (stored_unaliased_heap_gen fx Hnh is_local u T sets dominated
           to_unbox tcr e tenv canon statics statics' valuation valuation' β
           fuel m t n v v' Hfrag Hwfu Honce Hleast Hlocal Hsets Henv Hfragt
           Halloc Hst Hwhole Hcopies Hdomc Huses
           (TU.closures_unaliased_traversal_heap u e Hfrag Hwfu Honce)
           Ht Hrv Hat Hsat).
Qed.

Theorem rewrite_stored_type_sound_traversal_heap :
  rewrite_stored_type_sound_traversal_heap_statement.
Proof.
  exact (stored_traversal_heap_gen _ TP.rewrite_stored_type_sound_slots).
Qed.

Theorem rewrite_stored_type_sound_traversal_heap_landed :
  rewrite_stored_type_sound_traversal_heap_landed_statement.
Proof.
  exact (stored_traversal_heap_gen _
           TP.rewrite_stored_type_sound_slots_landed).
Qed.

Theorem stored_type_sound_traversal_heap :
  stored_type_sound_traversal_heap_statement.
Proof.
  exact (stored_unaliased_heap_gen _ TP.rewrite_stored_type_sound_slots).
Qed.

Theorem stored_type_sound_traversal_heap_landed :
  stored_type_sound_traversal_heap_landed_statement.
Proof.
  exact (stored_unaliased_heap_gen _
           TP.rewrite_stored_type_sound_slots_landed).
Qed.

Theorem stored_type_sound_full_heap : stored_type_sound_full_heap_statement.
Proof.
  exact (stored_full_heap_gen _ TP.rewrite_stored_type_sound_slots).
Qed.

Theorem stored_type_sound_full_heap_landed :
  stored_type_sound_full_heap_landed_statement.
Proof.
  exact (stored_full_heap_gen _ TP.rewrite_stored_type_sound_slots_landed).
Qed.

(* ======================================================================== *)
(* The exported types of the unit rebuilt with local unboxing               *)
(* ======================================================================== *)

Lemma unbox_types_gen (local loose : bool) (fx : fixes) :
  TP.fixes_sound fx -> unbox_types_end_to_end_heap_with local loose fx.
Proof.
  intros Hfx flag is_local u e T cannot_change code_deps dominated to_unbox
    b sets tenv canon valuation valuation' n s k vs hp rd sym allocated
    Hlocal Hleast Hfrag Hdc Hrk Hwf Hexits Honce Hvals Hsets Hunbox Hreads
    Hrun Hstep Henv Hfr Hcopies Hl.
  assert (Hcl : check_closed is_local (traverse_unit flag u e) T = true).
  { destruct local; cbn iota in Hleast.
    - exact (proj1 (FlambdaReaperLeastProofs.least_local_checks is_local _ T
                      Hleast)).
    - exact (proj1 (proj1 Hleast)). }
  assert (Hplain :
    ∃ (m : nat) (t : state) (vs' : list value) (hp' : heap)
      (rd' : read_record),
      run m
        (State
           (rebuild_unit_from_tables (traverse_unit flag u e) T b e)
           [] [] empty_static_env empty_heap no_reads)
      = Step t
      ∧ step t = Exit k vs' hp' rd'
      ∧ FlambdaReaperRebuildSpec.exits_observed_equal vs hp vs' hp'
      ∧ type_satisfied (state_statics t) valuation' (Value_symbol sym)
          (rewrite_symbol_type fx
             (plain_context is_local (traverse_unit flag u e) T sets) tenv
             canon sym)).
  { destruct local, loose; cbn iota in Hleast.
    - exact (THP.unaliased_types_end_to_end_term_heap_local_gen fx Hfx flag
               is_local u e T b sets tenv canon valuation valuation' n s k
               vs hp rd sym allocated Hlocal Hfrag Hdc Hrk Hwf Hexits Hleast
               Honce Hvals Hsets Hrun Hstep Henv Hfr Hcopies Hl).
    - exact (THP.types_end_to_end_term_heap_local_gen fx Hfx flag is_local u
               e T b sets tenv canon valuation valuation' n s k vs hp rd sym
               allocated Hlocal Hfrag Hdc Hrk Hwf Hexits Hleast Honce Hvals
               Hsets Hrun Hstep Henv Hfr Hcopies Hl).
    - destruct Hleast as [Hleast Hslots].
      exact (THP.unaliased_types_end_to_end_term_heap_gen fx Hfx flag
               is_local u e T b sets tenv canon valuation valuation' n s k
               vs hp rd sym allocated Hlocal Hslots Hfrag Hdc Hrk Hwf Hexits
               Hleast Honce Hvals Hsets Hrun Hstep Henv Hfr Hcopies Hl).
    - destruct Hleast as [Hleast Hslots].
      exact (THP.types_end_to_end_term_heap_gen fx Hfx flag is_local u e T b
               sets tenv canon valuation valuation' n s k vs hp rd sym
               allocated Hlocal Hslots Hfrag Hdc Hrk Hwf Hexits Hleast Honce
               Hvals Hsets Hrun Hstep Henv Hfr Hcopies Hl). }
  destruct Hplain as (m & t & vs' & hp' & rd' & Hm & Ht & Hobs & Hsat).
  destruct (THX.unbox_exit_heap flag is_local u e T cannot_change code_deps
              dominated to_unbox b m t k vs' hp' rd' Hlocal Hfrag Hcl Hunbox
              Hreads Hm Ht)
    as (m2 & t2 & rd2 & Hm2 & Ht2 & Hst2).
  exists m2, t2, vs', hp', rd2.
  split; [ exact Hm2 | split; [ exact Ht2 | split; [ exact Hobs | ] ] ].
  rewrite Hst2. exact Hsat.
Qed.

Theorem unbox_types_end_to_end_heap : unbox_types_end_to_end_heap_statement.
Proof. exact (unbox_types_gen false false _ TP.all_fixes_sound). Qed.

Theorem unbox_types_end_to_end_heap_landed :
  unbox_types_end_to_end_heap_landed_statement.
Proof. exact (unbox_types_gen false false _ TP.landed_fixes_sound). Qed.

Theorem unaliased_unbox_types_end_to_end_heap :
  unaliased_unbox_types_end_to_end_heap_statement.
Proof. exact (unbox_types_gen false true _ TP.all_fixes_sound). Qed.

Theorem unaliased_unbox_types_end_to_end_heap_landed :
  unaliased_unbox_types_end_to_end_heap_landed_statement.
Proof. exact (unbox_types_gen false true _ TP.landed_fixes_sound). Qed.

Theorem unbox_types_end_to_end_heap_local :
  unbox_types_end_to_end_heap_local_statement.
Proof. exact (unbox_types_gen true false _ TP.all_fixes_sound). Qed.

Theorem unbox_types_end_to_end_heap_local_landed :
  unbox_types_end_to_end_heap_local_landed_statement.
Proof. exact (unbox_types_gen true false _ TP.landed_fixes_sound). Qed.

Theorem unaliased_unbox_types_end_to_end_heap_local :
  unaliased_unbox_types_end_to_end_heap_local_statement.
Proof. exact (unbox_types_gen true true _ TP.all_fixes_sound). Qed.

Theorem unaliased_unbox_end_to_end_heap_local_landed :
  unaliased_unbox_end_to_end_heap_local_landed_statement.
Proof. exact (unbox_types_gen true true _ TP.landed_fixes_sound). Qed.

(* ======================================================================== *)
(* An example                                                               *)
(* ======================================================================== *)

Theorem unbox_types_heap_example : unbox_types_heap_example_statement.
Proof.
  do 8 (split; [ vm_compute; reflexivity | ]).
  split; [ intros f Hf; discriminate Hf | ].
  split.
  { apply (FlambdaReaperLeastProofs.check_least_sound _ _ 100).
    vm_compute. reflexivity. }
  split; [ intros f; reflexivity | ].
  split.
  { (* The graph has no constructor at a function slot. *)
    intros decls sites slot id Hl Hsites. exfalso.
    assert (Hb : is_bound slot decls = true)
      by (unfold is_bound; rewrite Hl; reflexivity).
    pose proof (proj2 (Hsites slot id Hl slot (sites slot))
                  (conj Hb eq_refl)) as Hcons.
    vm_compute in Hcons.
    repeat (destruct Hcons as [H | Hcons]; [ discriminate H | ]).
    exact Hcons. }
  repeat (split; [ vm_compute; reflexivity | ]).
  vm_compute. reflexivity.
Qed.

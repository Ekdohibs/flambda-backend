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
   statements of the traversal of a unit follow as without the heap.

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

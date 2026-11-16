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


(* Proofs of the statements of FlambdaTypeWellFormedSpec.v. Nothing here
   needs review. *)

From Stdlib Require Import ZArith List Bool Lia Setoid.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaValueSpec FlambdaSemanticsSpec
  FlambdaPrimitiveProofs.
From OxCamlFlambda2 Require FlambdaClosuresSpec FlambdaClosuresProofs.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeMeaningSpec FlambdaTypeMeaningProofs
  FlambdaTypeWellFormedSpec.
Import ListNotations.

(* ======================================================================== *)
(* The conditions on one structure                                          *)
(* ======================================================================== *)

Lemma non_empty_iff {A : Type} (l : list A) : non_empty l = true <-> l <> [].
Proof.
  destruct l as [ | a l]; simpl; split;
    [ discriminate | intros H; destruct (H eq_refl) | discriminate
    | reflexivity ].
Qed.

Theorem block_shape_canonical_b_iff : block_shape_canonical_b_iff_statement.
Proof.
  intros [[ | value_prefix_size flat_suffix] | ]; simpl;
    [ split; [ intros _; exact I | reflexivity ]
    | apply non_empty_iff
    | split; [ intros _; exact I | reflexivity ] ].
Qed.

Lemma block_product_fits_size (index : row_like_index nat block_shape)
    (maps_to : list flambda_type) :
  block_product_fits index maps_to = true ->
  length maps_to <= block_index_size index.
Proof.
  unfold block_product_fits, block_index_size.
  destruct (index_domain index) as [size | size].
  - rewrite Nat.eqb_eq. lia.
  - rewrite Nat.leb_le. lia.
Qed.

Lemma all_among_refl (names : list positive) :
  FlambdaMapSpec.all_among names names = true.
Proof. apply all_among. auto. Qed.

(* ======================================================================== *)
(* One equation for each structure                                          *)
(* ======================================================================== *)

(* What the tests come to on each constructor that holds a structure.
   They hold by definition. *)

Lemma non_null_value_well_formed (head : head_of_kind_value_non_null) :
  type_well_formed (non_null_value head)
  = head_of_kind_value_non_null_well_formed head.
Proof. reflexivity. Qed.

Lemma variant_well_formed (is_int : option variable)
    (immediates : Or_unknown.t flambda_type) (get_tag : option variable)
    (blocks : Or_unknown.t row_like_for_blocks)
    (extensions : variant_extensions) (is_unique : bool) :
  head_of_kind_value_non_null_well_formed
    (Variant is_int immediates get_tag blocks extensions is_unique)
  = match immediates with
    | Or_unknown.Known ty =>
      kind_eqb (kind ty) (Naked_number Kind_naked_immediate)
      && type_well_formed ty
    | Or_unknown.Unknown => true
    end
    && match blocks with
       | Or_unknown.Known blocks => row_like_for_blocks_well_formed blocks
       | Or_unknown.Unknown => true
       end
    && match extensions with
       | No_extensions => true
       | Ext when_immediate when_block =>
         env_extension_well_formed when_immediate
         && env_extension_well_formed when_block
       end.
Proof. reflexivity. Qed.

Lemma boxed_number_well_formed (k : boxable_number)
    (contents : flambda_type) (alloc_mode : Alloc_mode.t) :
  head_of_kind_value_non_null_well_formed (Boxed_number k contents alloc_mode)
  = kind_eqb (kind contents) (unboxed_kind k) && type_well_formed contents.
Proof. reflexivity. Qed.

Lemma row_like_for_blocks_well_formed_eq
    (known_tags : Keyed.finite_map Z (Or_unknown.t row_like_block_case))
    (other_tags : Or_bottom.t row_like_block_case)
    (alloc_mode : Alloc_mode.t) :
  row_like_for_blocks_well_formed
    (Row_like_for_blocks known_tags other_tags alloc_mode)
  = Keyed.distinct Z.eqb (keys known_tags)
    && Keyed.for_all_bindings_b Z.eqb
         (fun _ case =>
            match case with
            | Or_unknown.Known case => row_like_block_case_well_formed case
            | Or_unknown.Unknown => true
            end)
         known_tags
    && match other_tags with
       | Or_bottom.Ok case => row_like_block_case_well_formed case
       | Or_bottom.Bottom => true
       end.
Proof. reflexivity. Qed.

Lemma row_like_for_closures_well_formed_eq
    (known_closures : finite_map row_like_closures_case) :
  row_like_for_closures_well_formed (Row_like_for_closures known_closures)
  = distinct (keys known_closures)
    && for_all_bindings_b row_like_closures_case_well_formed known_closures.
Proof. reflexivity. Qed.

Lemma closures_entry_well_formed_eq
    (index : row_like_index set_of_closures_contents unit)
    (function_types : finite_map (Or_unknown.t function_type))
    (closure_types value_slot_types : finite_map flambda_type) :
  closures_entry_well_formed index
    (Closures_entry function_types closure_types value_slot_types)
  = closures_entry_slots_fit index (domain function_types)
      (domain closure_types) (domain value_slot_types)
    && distinct (keys function_types)
    && distinct (keys closure_types)
    && distinct (keys value_slot_types)
    && for_all_bindings_b
         (fun _ ty => kind_eqb (kind ty) Value && type_well_formed ty)
         closure_types
    && for_all_bindings_b (fun _ ty => type_well_formed ty) value_slot_types.
Proof. reflexivity. Qed.

Lemma env_extension_well_formed_eq
    (equations : Keyed.finite_map name flambda_type) :
  env_extension_well_formed (Env_extension equations)
  = Keyed.distinct name_eqb (keys equations)
    && Keyed.for_all_bindings_b name_eqb (fun _ ty => type_well_formed ty)
         equations.
Proof. reflexivity. Qed.

(* ======================================================================== *)
(* The structures that hold maps, by what the maps bind                     *)
(* ======================================================================== *)

Theorem row_like_block_case_well_formed_iff
  : row_like_block_case_well_formed_iff_statement.
Proof.
  intros case. unfold row_like_block_case_well_formed, block_case_well_formed.
  rewrite !andb_true_iff, (block_shape_canonical_b_iff _),
    (every_b_every flambda_type), (every_in flambda_type).
  tauto.
Qed.


Theorem row_like_for_blocks_well_formed_iff
  : row_like_for_blocks_well_formed_iff_statement.
Proof.
  intros known_tags other_tags alloc_mode.
  rewrite row_like_for_blocks_well_formed_eq, !andb_true_iff,
    (keyed_for_all_bindings_b Z Z.eqb Z.eqb_eq),
    (keyed_for_all_bindings_lookup Z Z.eqb Z.eqb_eq).
  split.
  - intros [[Hdistinct Hknown] Hother]. split; [ exact Hdistinct | ]. split.
    + intros tag case Hlookup. exact (Hknown tag _ Hlookup).
    + destruct other_tags as [case | ]; [ exact Hother | exact I ].
  - intros [Hdistinct [Hknown Hother]]. split; [ split | ].
    + exact Hdistinct.
    + intros tag [case | ] Hlookup;
        [ exact (Hknown tag case Hlookup) | reflexivity ].
    + destruct other_tags as [case | ]; [ exact Hother | reflexivity ].
Qed.


Theorem row_like_for_closures_well_formed_iff
  : row_like_for_closures_well_formed_iff_statement.
Proof.
  intros known_closures.
  rewrite row_like_for_closures_well_formed_eq, andb_true_iff,
    (keyed_for_all_bindings_b positive Pos.eqb Pos.eqb_eq),
    (for_all_bindings_lookup row_like_closures_case).
  reflexivity.
Qed.


Theorem closures_entry_well_formed_iff
  : closures_entry_well_formed_iff_statement.
Proof.
  intros index function_types closure_types value_slot_types.
  rewrite closures_entry_well_formed_eq, !andb_true_iff,
    !(keyed_for_all_bindings_b positive Pos.eqb Pos.eqb_eq flambda_type),
    !(for_all_bindings_lookup flambda_type).
  split.
  - intros [[[[[Hslots Hd1] Hd2] Hd3] Hclosures] Hvalue_slots].
    split; [ exact Hslots | ]. split; [ auto | ].
    split; [ | exact Hvalue_slots ]. intros g ty Hlookup.
    specialize (Hclosures g ty Hlookup). cbv beta in Hclosures.
    apply andb_true_iff in Hclosures as [Hkind Hty].
    split; [ apply kind_eqb_eq; exact Hkind | exact Hty ].
  - intros [Hslots [[Hd1 [Hd2 Hd3]] [Hclosures Hvalue_slots]]].
    split; [ split; [ auto | ] | exact Hvalue_slots ].
    intros g ty Hlookup. destruct (Hclosures g ty Hlookup) as [Hkind Hty].
    cbv beta. apply andb_true_iff.
    split; [ apply kind_eqb_eq; exact Hkind | exact Hty ].
Qed.


Theorem env_extension_well_formed_iff
  : env_extension_well_formed_iff_statement.
Proof.
  intros equations.
  rewrite env_extension_well_formed_eq, andb_true_iff,
    (keyed_for_all_bindings_b name name_eqb name_eqb_eq),
    (keyed_for_all_bindings_lookup name name_eqb name_eqb_eq).
  reflexivity.
Qed.


(* ======================================================================== *)
(* The creation functions give well-formed types                            *)
(* ======================================================================== *)

Theorem unknown_well_formed : unknown_well_formed_statement.
Proof. intros [ | [] | | ]; reflexivity. Qed.

Theorem bottom_well_formed : bottom_well_formed_statement.
Proof. intros [ | [] | | ]; reflexivity. Qed.

Theorem alias_type_of_well_formed : alias_type_of_well_formed_statement.
Proof. intros [ | [] | | ] s; reflexivity. Qed.

Theorem type_for_const_well_formed : type_for_const_well_formed_statement.
Proof. intros c. destruct c; try reflexivity. apply unknown_well_formed. Qed.

Theorem null_well_formed : null_well_formed_statement.
Proof. reflexivity. Qed.

Theorem any_non_null_value_well_formed
  : any_non_null_value_well_formed_statement.
Proof. reflexivity. Qed.

Theorem mutable_block_well_formed : mutable_block_well_formed_statement.
Proof. intros alloc_mode. reflexivity. Qed.

Theorem this_immutable_string_well_formed
  : this_immutable_string_well_formed_statement.
Proof. intros bytes. reflexivity. Qed.

Theorem these_naked_immediates_well_formed
  : these_naked_immediates_well_formed_statement.
Proof. intros [ | i [ | i' imms]]; reflexivity. Qed.

Theorem these_tagged_immediates_well_formed
  : these_tagged_immediates_well_formed_statement.
Proof. intros [ | i [ | i' imms]]; reflexivity. Qed.

Theorem any_tagged_immediate_well_formed
  : any_tagged_immediate_well_formed_statement.
Proof. reflexivity. Qed.

Theorem any_block_well_formed : any_block_well_formed_statement.
Proof. reflexivity. Qed.

Theorem relation_types_well_formed : relation_types_well_formed_statement.
Proof. intros n. repeat split; reflexivity. Qed.

Theorem create_variant_well_formed : create_variant_well_formed_statement.
Proof.
  intros is_unique immediates blocks extensions. unfold create_variant.
  rewrite non_null_value_well_formed, variant_well_formed, !andb_true_iff.
  destruct immediates as [ty | ], blocks as [blocks | ],
    extensions as [ | when_immediate when_block];
    rewrite ?andb_true_iff, ?(kind_eqb_eq _ _); intuition auto.
Qed.

Theorem tag_immediate_well_formed : tag_immediate_well_formed_statement.
Proof.
  intros t. unfold tag_immediate.
  rewrite
    (create_variant_well_formed false (Or_unknown.Known t)
       (Or_unknown.Known row_like_for_blocks_bottom) No_extensions).
  split; [ intros [Ht _]; exact Ht | ].
  intros Ht. split; [ exact Ht | ]. split; [ reflexivity | exact I ].
Qed.

Theorem box_number_well_formed : box_number_well_formed_statement.
Proof.
  intros k contents alloc_mode. unfold box_number.
  rewrite non_null_value_well_formed, boxed_number_well_formed,
    andb_true_iff, (kind_eqb_eq _ _).
  reflexivity.
Qed.

Theorem create_closures_well_formed : create_closures_well_formed_statement.
Proof. intros alloc_mode by_function_slot. reflexivity. Qed.

Lemma keys_of_unknown_cases {A : Type} (tags : list Z) :
  keys (map (fun tag => (tag, @Or_unknown.Unknown A)) tags) = tags.
Proof.
  unfold keys. rewrite map_map. simpl. apply map_id.
Qed.

Theorem blocks_with_these_tags_well_formed
  : blocks_with_these_tags_well_formed_statement.
Proof.
  intros tags alloc_mode. unfold blocks_with_these_tags.
  rewrite
    (create_variant_well_formed false
       (Or_unknown.Known (bottom (Naked_number Kind_naked_immediate)))
       (Or_unknown.Known
          (Row_like_for_blocks
             (map (fun tag => (tag, Or_unknown.Unknown)) tags)
             Or_bottom.Bottom alloc_mode))
       No_extensions),
    (row_like_for_blocks_well_formed_iff _ _ _), keys_of_unknown_cases.
  split.
  - intros [_ [[Hdistinct _] _]]. exact Hdistinct.
  - intros Hdistinct. split; [ split; reflexivity | ]. split; [ | exact I ].
    split; [ exact Hdistinct | ]. split; [ | exact I ].
    intros tag case. rewrite lookup_unknown_cases.
    destruct (existsb (Z.eqb tag) tags); discriminate.
Qed.

(* Blocks of one case. *)

Lemma block_case_of_parts_well_formed
    (domain : Row_like_index_domain.t nat) (shape : block_shape)
    (maps_to : list flambda_type) :
  row_like_block_case_well_formed
    (Row_like_case maps_to (Row_like_index domain shape) empty_env_extension)
  = true
  <-> block_product_fits (Row_like_index domain shape) maps_to = true
      ∧ block_shape_canonical shape
      ∧ (∀ ty : flambda_type, ty ∈ maps_to -> type_well_formed ty = true).
Proof.
  rewrite (row_like_block_case_well_formed_iff _). simpl.
  split; [ intros [Hfits [Hshape [Htypes _]]]; auto | ].
  intros [Hfits [Hshape Htypes]]. auto.
Qed.

Lemma one_tag_well_formed (tag : Z) (case : row_like_block_case)
    (alloc_mode : Alloc_mode.t) :
  row_like_for_blocks_well_formed
    (Row_like_for_blocks [(tag, Or_unknown.Known case)] Or_bottom.Bottom
       alloc_mode)
  = true
  <-> row_like_block_case_well_formed case = true.
Proof.
  rewrite (row_like_for_blocks_well_formed_iff _ _ _). simpl. split.
  - intros [_ [Hknown _]]. apply (Hknown tag). rewrite Z.eqb_refl.
    reflexivity.
  - intros Hcase. split; [ reflexivity | ]. split; [ | exact I ].
    intros tag' case'. destruct (Z.eqb tag' tag); [ | discriminate ].
    intros [= <-]. exact Hcase.
Qed.

Lemma other_tags_well_formed (case : row_like_block_case)
    (alloc_mode : Alloc_mode.t) :
  row_like_for_blocks_well_formed
    (Row_like_for_blocks [] (Or_bottom.Ok case) alloc_mode)
  = true
  <-> row_like_block_case_well_formed case = true.
Proof.
  rewrite (row_like_for_blocks_well_formed_iff _ _ _). simpl. split.
  - intros [_ [_ Hcase]]. exact Hcase.
  - intros Hcase. split; [ reflexivity | ]. split; [ | exact Hcase ].
    intros tag case'. discriminate.
Qed.

Theorem row_like_for_blocks_create_well_formed
  : row_like_for_blocks_create_well_formed_statement.
Proof.
  intros tag size shape maps_to alloc_mode. cbv zeta.
  unfold row_like_for_blocks_create_exactly,
    row_like_for_blocks_create_at_least,
    row_like_for_blocks_create_at_least_unknown_tag.
  rewrite !one_tag_well_formed, other_tags_well_formed,
    !block_case_of_parts_well_formed.
  unfold block_product_fits. simpl. rewrite Nat.eqb_eq, Nat.leb_le.
  split; [ reflexivity | ]. split; reflexivity.
Qed.

Theorem immutable_block_well_formed : immutable_block_well_formed_statement.
Proof.
  intros is_unique tag shape alloc_mode fields. unfold immutable_block.
  rewrite
    (create_variant_well_formed is_unique
       (Or_unknown.Known (bottom (Naked_number Kind_naked_immediate)))
       (Or_unknown.Known
          (row_like_for_blocks_create_exactly tag (length fields) shape
             fields alloc_mode))
       No_extensions).
  destruct
    (row_like_for_blocks_create_well_formed tag (length fields) shape fields
       alloc_mode)
    as [Hexactly _].
  cbv zeta in Hexactly. rewrite Hexactly. split.
  - intros [_ [[_ Hfields] _]]. exact Hfields.
  - intros Hfields. split; [ split; reflexivity | ].
    split; [ split; [ reflexivity | exact Hfields ] | exact I ].
Qed.

(* Closures of one case. *)

Lemma one_slot_well_formed (slot : function_slot)
    (case : row_like_closures_case) :
  row_like_for_closures_well_formed (Row_like_for_closures [(slot, case)])
  = true
  <-> row_like_closures_case_well_formed slot case = true.
Proof.
  rewrite (row_like_for_closures_well_formed_iff _). simpl. split.
  - intros [_ Hknown]. apply Hknown. rewrite Pos.eqb_refl. reflexivity.
  - intros Hcase. split; [ reflexivity | ]. intros slot' case'.
    destruct (Pos.eqb slot' slot) eqn:Hslot; [ | discriminate ].
    apply Pos.eqb_eq in Hslot as ->. intros [= <-]. exact Hcase.
Qed.

Theorem exactly_this_closure_well_formed
  : exactly_this_closure_well_formed_statement.
Proof.
  intros this_function_slot function_types closure_types value_slot_types
    alloc_mode.
  unfold exactly_this_closure.
  rewrite (create_closures_well_formed _ _), one_slot_well_formed.
  unfold row_like_closures_case_well_formed. simpl case_index.
  simpl case_maps_to. simpl case_env_extension.
  rewrite !andb_true_iff, (closures_entry_well_formed_iff _ _ _ _).
  unfold closures_index_has_slot, closures_entry_slots_fit,
    closures_index_contents.
  simpl. rewrite (among_domain _), !andb_true_iff, !all_among_refl.
  split.
  - intros [[Hslot [[[[_ Hclosure_slots] _] _] Hrest]] _]. auto.
  - intros [Hslot [Hclosure_slots Hrest]].
    split; [ | reflexivity ]. split; [ exact Hslot | ].
    split; [ | exact Hrest ].
    split; [ | reflexivity ]. split; [ | reflexivity ].
    split; [ reflexivity | exact Hclosure_slots ].
Qed.

Theorem at_least_this_closure_well_formed
  : at_least_this_closure_well_formed_statement.
Proof.
  intros this_function_slot function_types closure_types value_slot_types
    alloc_mode.
  unfold at_least_this_closure.
  rewrite (create_closures_well_formed _ _), one_slot_well_formed.
  unfold row_like_closures_case_well_formed. simpl case_index.
  simpl case_maps_to. simpl case_env_extension.
  rewrite !andb_true_iff, (closures_entry_well_formed_iff _ _ _ _).
  unfold closures_index_has_slot, closures_entry_slots_fit,
    closures_index_contents.
  simpl. rewrite !andb_true_iff, !all_among_refl.
  split.
  - intros [[_ [[[[_ Hclosure_slots] _] _] Hrest]] _]. auto.
  - intros [Hclosure_slots Hrest].
    split; [ | reflexivity ]. split; [ reflexivity | ].
    split; [ | exact Hrest ].
    split; [ | reflexivity ]. split; [ | reflexivity ].
    split; [ reflexivity | exact Hclosure_slots ].
Qed.

(* ======================================================================== *)
(* What the conditions give                                                 *)
(* ======================================================================== *)

Theorem well_formed_block_case_empty : well_formed_block_case_empty_statement.
Proof.
  intros statics vars tag fields case ty Hcase.
  apply block_case_empty.
  apply (row_like_block_case_well_formed_iff case) in Hcase as [Hfits _].
  exact (block_product_fits_size _ _ Hfits).
Qed.

Theorem well_formed_block_cases_shape
  : well_formed_block_cases_shape_statement.
Proof.
  intros statics vars tag fields case case' Hcase Hcase'
    [[_ Hshape] _] [[_ Hshape'] _].
  apply (row_like_block_case_well_formed_iff case) in Hcase
    as [_ [Hcanonical _]].
  apply (row_like_block_case_well_formed_iff case') in Hcase'
    as [_ [Hcanonical' _]].
  exact
    (block_shapes_disjoint _ _ tag fields Hcanonical Hcanonical' Hshape
       Hshape').
Qed.

(* A slot that a map of an entry binds is among those of the index, if
   the slots of the map are. *)
Lemma bound_among {A : Type} (name : positive) (a : A)
    (bindings : finite_map A) (allowed : list positive) :
  FlambdaMapSpec.all_among (domain bindings) allowed = true ->
  lookup name bindings = Some a ->
  among name allowed = true.
Proof.
  intros Hall Hlookup. apply (proj1 (all_among _ _) Hall).
  rewrite (among_domain A). apply is_bound_lookup. exists a. exact Hlookup.
Qed.

Theorem well_formed_closures_case_function_code
  : well_formed_closures_case_function_code_statement.
Proof.
  intros statics vars v slot functions value_slots function_types
    closure_types value_slot_types index extension g function_type.
  cbv zeta. unfold row_like_closures_case_well_formed.
  simpl case_index. simpl case_maps_to. simpl case_env_extension.
  rewrite !andb_true_iff, (closures_entry_well_formed_iff _ _ _ _).
  intros [[_ [Hslots _]] _] Hsatisfied Hlookup.
  apply
    (closures_case_function_code statics vars v slot functions value_slots
       function_types closure_types value_slot_types index extension g
       function_type Hsatisfied);
    [ | exact Hlookup ].
  unfold closures_entry_slots_fit in Hslots.
  rewrite !andb_true_iff in Hslots.
  destruct Hslots as [[[Hfunction_slots _] _] _].
  exact (bound_among _ _ _ _ Hfunction_slots Hlookup).
Qed.

Theorem well_formed_closures_case_value_slot
  : well_formed_closures_case_value_slot_statement.
Proof.
  intros statics vars v slot functions value_slots function_types
    closure_types value_slot_types index extension w ty.
  cbv zeta. unfold row_like_closures_case_well_formed.
  simpl case_index. simpl case_maps_to. simpl case_env_extension.
  rewrite !andb_true_iff, (closures_entry_well_formed_iff _ _ _ _).
  intros [[_ [Hslots _]] _] Hsatisfied Hlookup.
  apply
    (closures_case_value_slot statics vars v slot functions value_slots
       function_types closure_types value_slot_types index extension w ty
       Hsatisfied);
    [ | exact Hlookup ].
  unfold closures_entry_slots_fit in Hslots.
  rewrite !andb_true_iff in Hslots.
  destruct Hslots as [[[_ _] Hvalue_slots] _].
  exact (bound_among _ _ _ _ Hvalue_slots Hlookup).
Qed.

(* A function slot among the closures of the index of a case that
   admits a closure is a function of the set of the closure. *)
Lemma closures_index_function_bound (statics : static_env) (vars : env)
    (v : value) (slot : function_slot)
    (functions : list (function_slot * code_id))
    (value_slots : list (value_slot * value))
    (case : row_like_closures_case) (g : function_slot) :
  row_like_closures_case_satisfied statics vars v slot functions value_slots
    case ->
  among g (contents_closures (closures_index_contents (case_index case)))
  = true ->
  is_bound g functions = true.
Proof.
  intros [Hindex _] Hamong. rewrite <- among_domain.
  unfold closures_index_satisfied in Hindex.
  unfold closures_index_contents in Hamong.
  destruct (index_domain (case_index case)) as [contents | contents].
  - destruct Hindex as [Hsame _]. rewrite <- (Hsame g). exact Hamong.
  - destruct Hindex as [Hall _].
    exact (proj1 (all_among _ _) Hall g Hamong).
Qed.

Theorem well_formed_closures_case_closure_type
  : well_formed_closures_case_closure_type_statement.
Proof.
  intros statics vars v slot functions value_slots function_types
    closure_types value_slot_types index extension g ty.
  cbv zeta. intros Hvalid Hclosure Hwell_formed Hsatisfied Hlookup.
  unfold row_like_closures_case_well_formed in Hwell_formed.
  simpl case_index in Hwell_formed. simpl case_maps_to in Hwell_formed.
  simpl case_env_extension in Hwell_formed.
  rewrite !andb_true_iff, (closures_entry_well_formed_iff _ _ _ _)
    in Hwell_formed.
  destruct Hwell_formed as [[_ [Hslots _]] _].
  unfold closures_entry_slots_fit in Hslots.
  rewrite !andb_true_iff in Hslots.
  destruct Hslots as [[[_ Hclosure_slots] _] _].
  pose proof (bound_among _ _ _ _ Hclosure_slots Hlookup) as Hamong.
  pose proof (closures_index_function_bound statics vars v slot functions
                value_slots _ g Hsatisfied Hamong)
    as Hbound.
  destruct
    (FlambdaClosuresProofs.project_function_slot_valid statics slot g v
       functions value_slots Hvalid Hclosure Hbound)
    as [closure Hproject].
  exists closure. split; [ exact Hproject | ].
  destruct Hsatisfied as [_ [Hentry _]]. simpl in Hentry.
  apply closures_entry_lookup_iff in Hentry as [_ [Hclosures _]].
  exact (Hclosures g ty closure Hlookup Hproject).
Qed.

Theorem well_formed_exact_case_function_type
  : well_formed_exact_case_function_type_statement.
Proof.
  intros slot function_types closure_types value_slot_types contents
    extension.
  unfold row_like_closures_case_well_formed.
  simpl case_index. simpl case_maps_to. simpl case_env_extension.
  rewrite !andb_true_iff, (closures_entry_well_formed_iff _ _ _ _).
  intros [[Hslot [Hslots _]] _].
  unfold closures_entry_slots_fit in Hslots. simpl in Hslots.
  rewrite !andb_true_iff in Hslots. destruct Hslots as [_ Hexact].
  rewrite <- (among_domain _).
  exact (proj1 (all_among _ _) Hexact slot Hslot).
Qed.

(* ======================================================================== *)
(* Examples                                                                 *)
(* ======================================================================== *)

(* For each condition, a type that breaks it; and, of the types of the
   examples of the meaning (FlambdaTypeMeaningProofs.v), some that the
   creation functions make, which are well formed, and those that they
   do not make. *)
Module Well_formed_examples.

Import FlambdaTypeMeaningProofs.Examples.

Definition well_formed (t : flambda_type) : Prop := type_well_formed t = true.

Example mixed_block_type_well_formed : well_formed mixed_block_type.
Proof. reflexivity. Qed.

Example float_block_type_well_formed : well_formed (float_block_type 1).
Proof. reflexivity. Qed.

Example block_of_size_at_least_well_formed
  : well_formed (block_of_size_at_least 3).
Proof. reflexivity. Qed.

Example closure_of_unknown_code_well_formed
  : well_formed closure_of_unknown_code.
Proof. reflexivity. Qed.

(* 1: a set of numbers, and of strings, that is empty. *)
Example empty_numbers_not_well_formed
  : ~ well_formed (Type_naked_number Kind_naked_float (Ok (No_alias []))).
Proof. discriminate. Qed.

Example empty_immediates_not_well_formed
  : ~ well_formed
        (Type_naked_immediate
           (Ok
              (No_alias
                 (Head_of_kind_naked_immediate (Or_unknown.Known []) [])))).
Proof. discriminate. Qed.

Example empty_strings_not_well_formed
  : ~ well_formed (non_null_value (String [])).
Proof. discriminate. Qed.

(* 2. *)
Example naked_immediate_number_not_well_formed
  : ~ well_formed (Type_naked_number Kind_naked_immediate Unknown).
Proof. discriminate. Qed.

(* 3: immediates that are not of kind naked immediate. *)
Example immediates_of_values_not_well_formed
  : ~ well_formed (tag_immediate (unknown Value)).
Proof. discriminate. Qed.

(* 4: a boxed float whose contents are an [int32]. *)
Example boxed_float_of_int32_not_well_formed
  : ~ well_formed
        (box_number Boxable_naked_float
           (unknown (Naked_number Kind_naked_int32)) Alloc_mode.Heap).
Proof. discriminate. Qed.

(* 5: a closure type that is not of kind value. *)
Example closure_type_of_float_not_well_formed
  : ~ well_formed
        (exactly_this_closure unnamed_function_slot
           [(unnamed_function_slot, Or_unknown.Unknown)]
           [(unnamed_function_slot, unknown (Naked_number Kind_naked_float))]
           [] Alloc_mode.Heap).
Proof. discriminate. Qed.

(* 6: the product wider than the size of its index, which a block of
   two fields has ([wide_product_has]). *)
Example wide_product_not_well_formed
  : ~ well_formed
        (create_variant false
           (Or_unknown.Known (bottom (Naked_number Kind_naked_immediate)))
           (Or_unknown.Known
              (row_like_for_blocks_create_at_least 0 2 (Scannable Value_only)
                 [unknown Value; unknown Value; bottom Value]
                 Alloc_mode.Heap))
           No_extensions).
Proof. discriminate. Qed.

(* 7: the mixed record with no flat suffix, which a block of values has
   ([mixed_record_of_values_has]). *)
Example mixed_record_of_values_not_well_formed
  : ~ well_formed
        (immutable_block false 0 (Scannable (Mixed_record 2 []))
           Alloc_mode.Heap [unknown Value; unknown Value]).
Proof. discriminate. Qed.

(* 8a: the two types that the n-way join made before f489f9445e, with
   a function type of a slot that the index has not. *)
Example join_of_closure_types_not_well_formed
  : ~ well_formed (join_of_closure_types some_code).
Proof. discriminate. Qed.

Example nway_join_not_well_formed : ~ well_formed nway_join_with_known_code.
Proof. discriminate. Qed.

(* 8b: an exact set of closures that has not the slot of the closure. *)
Example exact_without_slot_not_well_formed
  : ~ well_formed
        (exactly_this_closure unnamed_function_slot [] [] [] Alloc_mode.Heap).
Proof. discriminate. Qed.

(* 8c: an exact index with a closure of which the entry has no function
   type. *)
Example exact_without_function_type_not_well_formed
  : ~ well_formed
        (create_closures Alloc_mode.Heap
           (Row_like_for_closures
              [(unnamed_function_slot,
                Row_like_case (Closures_entry [] [] [])
                  (Row_like_index
                     (Row_like_index_domain.Known
                        (Set_of_closures_contents [unnamed_function_slot] []))
                     tt)
                  empty_env_extension)])).
Proof. discriminate. Qed.

(* 9: an environment extension with two equations on one name. *)
Example two_equations_not_well_formed
  : ~ well_formed
        (create_variant false Or_unknown.Unknown Or_unknown.Unknown
           (Ext
              (Env_extension
                 [(Name_var x, unknown Value); (Name_var x, bottom Value)])
              empty_env_extension)).
Proof. discriminate. Qed.

End Well_formed_examples.

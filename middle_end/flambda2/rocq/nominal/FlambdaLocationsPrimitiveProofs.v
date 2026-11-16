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

(* Proofs of FlambdaLocationsPrimitiveSpec.v. This file needs no review:
   every statement it proves is in the specification. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaNominalSpec
  FlambdaNominalProofs FlambdaAlphaProofs FlambdaNameAbstractionSpec
  FlambdaNameAbstractionProofs FlambdaHeapSpec FlambdaHeapProofs
  FlambdaLocationsSpec FlambdaLocationsProofs FlambdaLocationsPrimitiveSpec.
From OxCamlFlambda2 Require FlambdaPrimitiveProofs.
Import ListNotations.

(* ======================================================================== *)
(* The relation between a value and the value moved                         *)
(* ======================================================================== *)

Definition moved (p : permutation) (v v' : value) : Prop :=
  v' = permute_value p v.

Lemma Forall2_moved (p : permutation) (vs vs' : list value) :
  Forall2 (moved p) vs vs' <-> vs' = map (permute_value p) vs.
Proof.
  split.
  - intros Hall. induction Hall as [ | v v' vs vs' Hv _ IH];
      [ reflexivity | ].
    simpl. rewrite Hv, IH. reflexivity.
  - intros ->. induction vs as [ | v vs IH]; constructor;
      [ reflexivity | exact IH ].
Qed.

Lemma lookup_permute_statics (p : permutation)
    (values : finite_map statically_allocated_value) (s : symbol) :
  lookup s (map_values (permute_static_value p) values)
  = option_map (permute_static_value p) (lookup s values).
Proof. apply lookup_map_values. Qed.

Ltac statics_lookup :=
  cbn [permute_static_env static_values]; rewrite lookup_permute_statics.

Lemma moved_respects (p : permutation) (statics : static_env) :
  respects_observations (moved p) statics (permute_static_env p statics).
Proof.
  split; [ | split ].
  - split.
    + intros c v' ->. reflexivity.
    + intros v c Hv. unfold moved in Hv.
      destruct v; simpl in Hv; try discriminate.
      injection Hv as ->. reflexivity.
  - split; [ | split ].
    + intros c. split; reflexivity.
    + intros tag fields fields' Hfields. apply Forall2_moved in Hfields.
      subst fields'. reflexivity.
    + intros elements elements' Helements.
      apply Forall2_moved in Helements. subst elements'. reflexivity.
  - intros v v' ->.
    split;
      [ | split; [ | split; [ | split; [ | split; [ | split; [ | split ]]]]]].
    + intros tag fields Hblock.
      destruct v; simpl in Hblock; try discriminate.
      * simpl. statics_lookup.
        destruct (lookup s (static_values statics))
          as [[ | | tag0 fields0 | | | | ] | ]; try discriminate.
        injection Hblock as <- <-. simpl. eexists.
        split; [ reflexivity | apply Forall2_moved; reflexivity ].
      * injection Hblock as <- <-. simpl. eexists.
        split; [ reflexivity | apply Forall2_moved; reflexivity ].
    + intros c Hboxed.
      destruct v; simpl in Hboxed |- *; try discriminate.
      * statics_lookup.
        destruct (lookup s (static_values statics))
          as [[ | | | contents | | | ] | ]; try discriminate.
        simpl. destruct contents; try discriminate. exact Hboxed.
      * exact Hboxed.
    + intros bytes Hstring.
      destruct v; simpl in Hstring |- *; try discriminate.
      statics_lookup.
      destruct (lookup s (static_values statics))
        as [[ | | | | bytes0 | | ] | ]; try discriminate.
      exact Hstring.
    + destruct v; simpl; try (intros Hclosure; exact Hclosure).
      * discriminate.
      * destruct (lookup slot function_decls); simpl;
          [ discriminate | intros Hclosure; exact Hclosure ].
      * statics_lookup.
        destruct (lookup s (static_values statics))
          as [[code slots | slot syms decls slots | | | | | ] | ]; simpl;
          try (intros Hclosure; exact Hclosure).
        -- discriminate.
        -- destruct (lookup slot decls); simpl;
             [ discriminate | intros Hclosure; exact Hclosure ].
    + destruct v; simpl; try (intros Hpartial; exact Hpartial).
      discriminate.
    + destruct v; simpl; try reflexivity.
      statics_lookup.
      destruct (lookup s (static_values statics)); simpl;
        [ reflexivity | discriminate ].
    + intros imported Himported.
      destruct v; simpl in Himported |- *; try discriminate.
      statics_lookup.
      destruct (lookup s (static_values statics))
        as [[ | | | | | | class] | ]; try discriminate.
      exact Himported.
    + intros elements Harray.
      destruct v; simpl in Harray |- *; try discriminate.
      * statics_lookup.
        destruct (lookup s (static_values statics))
          as [[ | | | | | elements0 | ] | ]; try discriminate.
        injection Harray as <-. simpl. eexists.
        split; [ reflexivity | apply Forall2_moved; reflexivity ].
      * injection Harray as <-. eexists.
        split; [ reflexivity | apply Forall2_moved; reflexivity ].
Qed.

Lemma eval_prim_moved (p : permutation) (statics : static_env) (prim : prim)
    (vs : list value) (v : value) :
  eval_prim statics prim vs = Some v ->
  eval_prim (permute_static_env p statics) prim (map (permute_value p) vs)
  = Some (permute_value p v).
Proof.
  intros Heval.
  destruct
    (FlambdaPrimitiveProofs.eval_prim_related (moved p) statics
       (permute_static_env p statics) prim vs (map (permute_value p) vs) v
       (moved_respects p statics))
    as (v' & Heval' & ->);
    [ apply Forall2_moved; reflexivity | exact Heval | exact Heval' ].
Qed.

Theorem eval_prim_permuted : eval_prim_permuted_statement.
Proof.
  intros p statics prim vs.
  destruct (eval_prim statics prim vs) as [v | ] eqn:Heval; simpl.
  - exact (eval_prim_moved p statics prim vs v Heval).
  - destruct
      (eval_prim (permute_static_env p statics) prim
         (map (permute_value p) vs))
      as [v' | ] eqn:Heval'; [ | reflexivity ].
    apply (eval_prim_moved (inverse_permutation p)) in Heval'.
    pose proof (proj1 (act_inverse _ _ p statics static_env_laws)) as Hst.
    pose proof (proj1 (act_inverse _ _ p vs values_laws)) as Hvs.
    cbv beta in Hst, Hvs. rewrite Hst, Hvs in Heval'. congruence.
Qed.

(* ======================================================================== *)
(* What the primitives of the heap are made of                              *)
(* ======================================================================== *)

Lemma forallb_map_moved {A : Type} (f g : A -> bool) (act : A -> A)
    (l : list A) :
  (∀ x : A, f (act x) = g x) -> forallb f (map act l) = forallb g l.
Proof.
  intros Hf. induction l as [ | x l IH]; simpl; [ reflexivity | ].
  rewrite Hf, IH. reflexivity.
Qed.

Lemma int_of_value_permute (p : permutation) (kind : standard_int)
    (v : value) :
  int_of_value kind (permute_value p v) = int_of_value kind v.
Proof. destruct v; try reflexivity; destruct kind; reflexivity. Qed.

Lemma is_tagged_immediate_permute (p : permutation) (v : value) :
  is_tagged_immediate (permute_value p v) = is_tagged_immediate v.
Proof.
  unfold is_tagged_immediate. rewrite int_of_value_permute. reflexivity.
Qed.

Lemma array_element_ok_permute (p : permutation) (kind : array_kind)
    (v : value) :
  array_element_ok kind (permute_value p v) = array_element_ok kind v.
Proof.
  destruct kind; simpl;
    first
      [ apply is_tagged_immediate_permute
      | rewrite kind_of_permute_value; reflexivity ].
Qed.

Lemma array_made_permute (p : permutation) (kind : array_kind)
    (vs : list value) :
  array_made kind (map (permute_value p) vs)
  = option_map (map (permute_value p)) (array_made kind vs).
Proof.
  unfold array_made.
  rewrite (forallb_map_moved _ (array_element_ok kind))
    by (intros v; apply array_element_ok_permute).
  destruct (forallb (array_element_ok kind) vs); reflexivity.
Qed.

Lemma array_index_permute (p : permutation) (elements : list value)
    (index : value) :
  array_index (map (permute_value p) elements) (permute_value p index)
  = array_index elements index.
Proof.
  unfold array_index. rewrite int_of_value_permute, length_map.
  reflexivity.
Qed.

Lemma array_loaded_permute (p : permutation) (load_kind : array_load_kind)
    (elements : list value) (index : value) :
  array_loaded load_kind (map (permute_value p) elements)
    (permute_value p index)
  = option_map (permute_value p) (array_loaded load_kind elements index).
Proof.
  unfold array_loaded. rewrite array_index_permute.
  destruct (array_index elements index) as [i | ]; simpl; [ | reflexivity ].
  rewrite nth_error_map.
  destruct (nth_error elements i) as [v | ]; simpl; [ | reflexivity ].
  rewrite array_element_ok_permute.
  destruct (array_element_ok _ v); reflexivity.
Qed.

Lemma replace_nth_map {A : Type} (f : A -> A) (n : nat) (x : A)
    (l : list A) :
  replace_nth n (f x) (map f l) = option_map (map f) (replace_nth n x l).
Proof.
  revert l. induction n as [ | n IH]; intros [ | y l]; simpl;
    try reflexivity.
  rewrite IH. destruct (replace_nth n x l); reflexivity.
Qed.

Lemma array_stored_permute (p : permutation) (set_kind : array_set_kind)
    (kind : array_kind) (elements : list value) (index new : value) :
  array_stored set_kind kind (map (permute_value p) elements)
    (permute_value p index) (permute_value p new)
  = option_map (map (permute_value p))
      (array_stored set_kind kind elements index new).
Proof.
  unfold array_stored. rewrite !array_element_ok_permute, array_index_permute.
  destruct (_ && _); [ | reflexivity ].
  destruct (array_index elements index) as [i | ]; simpl; [ | reflexivity ].
  apply replace_nth_map.
Qed.

Lemma array_length_of_permute (p : permutation) (elements : list value) :
  array_length_of (map (permute_value p) elements)
  = option_map (permute_value p) (array_length_of elements).
Proof.
  unfold array_length_of. rewrite length_map.
  destruct (in_range _ _); reflexivity.
Qed.

Lemma load_field_permute (p : permutation) (kind : block_access_kind)
    (field : Z) (tag : Z) (fields : list value) :
  load_field kind field tag (map (permute_value p) fields)
  = option_map (permute_value p) (load_field kind field tag fields).
Proof.
  destruct kind as [known_tag size field_kind]. simpl. rewrite length_map.
  destruct (_ && _); [ | reflexivity ].
  rewrite nth_error_map.
  destruct (nth_error fields (Z.to_nat field)) as [v | ]; simpl;
    [ | reflexivity ].
  rewrite kind_of_permute_value, is_tagged_immediate_permute.
  destruct
    match field_kind with
    | Any_value => kind_eqb (kind_of_value v) Value
    | Immediate => is_tagged_immediate v
    end;
    reflexivity.
Qed.

Lemma store_field_permute (p : permutation) (kind : block_access_kind)
    (field : Z) (new : value) (tag : Z) (fields : list value) :
  store_field kind field (permute_value p new) tag
    (map (permute_value p) fields)
  = option_map (map (permute_value p))
      (store_field kind field new tag fields).
Proof.
  destruct kind as [known_tag size field_kind]. simpl. rewrite length_map.
  rewrite kind_of_permute_value, is_tagged_immediate_permute.
  destruct (_ && _); [ | reflexivity ].
  apply replace_nth_map.
Qed.

Lemma exec_unary_on_cell_permute (p : permutation) (u : unary_primitive)
    (l : location) (c : cell) :
  exec_unary_on_cell u (permute p l) (permute_cell p c)
  = option_map (permute_value p) (exec_unary_on_cell u l c).
Proof.
  destruct c as [mut tag fields | kind elements]; destruct u; simpl;
    try reflexivity;
    first
      [ destruct (load_allowed _ _);
        [ apply load_field_permute | reflexivity ]
      | destruct (kind_eqb _ _); reflexivity
      | apply array_length_of_permute
      | destruct variant_only; reflexivity ].
Qed.

Lemma exec_binary_on_cell_permute (p : permutation) (b : binary_primitive)
    (l : location) (c : cell) (v2 : value) :
  exec_binary_on_cell b (permute p l) (permute_cell p c) (permute_value p v2)
  = option_map (permute_value p) (exec_binary_on_cell b l c v2).
Proof.
  destruct c as [mut tag fields | kind elements]; destruct b; simpl;
    try reflexivity.
  destruct (load_allowed _ _); [ apply array_loaded_permute | reflexivity ].
Qed.

Lemma location_of_permute (p : permutation) (v : value) :
  location_of (permute_value p v) = option_map (permute p) (location_of v).
Proof. destruct v; reflexivity. Qed.

Lemma is_location_permute (p : permutation) (v : value) :
  is_location (permute_value p v) = is_location v.
Proof. destruct v; reflexivity. Qed.

Lemma is_allocated_permute (p : permutation) (statics : static_env)
    (v : value) :
  is_allocated (permute_static_env p statics) (permute_value p v)
  = is_allocated statics v.
Proof.
  unfold is_allocated. destruct v; simpl; try reflexivity.
  - destruct (lookup _ _); reflexivity.
  - statics_lookup.
    destruct (lookup s (static_values statics))
      as [[code slots | slot syms decls slots | | contents | | | ] | ];
      simpl;
      try reflexivity.
    + destruct (lookup slot decls); reflexivity.
    + destruct contents; reflexivity.
Qed.

Lemma is_imported_pointer_permute (p : permutation) (statics : static_env)
    (v : value) :
  is_imported_pointer (permute_static_env p statics) (permute_value p v)
  = is_imported_pointer statics v.
Proof.
  unfold is_imported_pointer. destruct v; simpl; try reflexivity.
  rewrite lookup_permute_statics.
  destruct (lookup s (static_values statics))
    as [[ | | | | | | class] | ];
    reflexivity.
Qed.

Lemma eval_phys_equal_location_permute (p : permutation)
    (statics : static_env) (v1 v2 : value) :
  eval_phys_equal_location (permute_static_env p statics)
    (permute_value p v1) (permute_value p v2)
  = eval_phys_equal_location statics v1 v2.
Proof.
  rewrite !eval_phys_equal_location_cases, !location_of_permute,
    !int_of_value_permute, !is_allocated_permute,
    !is_imported_pointer_permute.
  destruct (location_of v1) as [l1 | ], (location_of v2) as [l2 | ]; simpl;
    try reflexivity.
  rewrite permute_eqb. reflexivity.
Qed.

Lemma argument_bound_permute (p : permutation) (statics : static_env)
    (h : heap) (v : value) :
  argument_bound (permute_static_env p statics) (permute_heap p h)
    (permute_value p v)
  = argument_bound statics h v.
Proof.
  destruct v; simpl; try reflexivity.
  - unfold is_bound. statics_lookup.
    destruct (lookup s (static_values statics)); reflexivity.
  - apply is_bound_permute_heap.
Qed.

Lemma arguments_bound_permute (p : permutation) (statics : static_env)
    (h : heap) (vs : list value) :
  forallb (argument_bound (permute_static_env p statics) (permute_heap p h))
    (map (permute_value p) vs)
  = forallb (argument_bound statics h) vs.
Proof.
  apply forallb_map_moved. intros v. apply argument_bound_permute.
Qed.

(* ======================================================================== *)
(* A primitive that does not bind a location                                *)
(* ======================================================================== *)

Lemma pure_permuted (p : permutation) (statics : static_env) (h : heap)
    (prim : prim) (vs : list value) :
  (let* v :=
     eval_prim (permute_static_env p statics) prim
       (map (permute_value p) vs)
   in
   Some (v, permute_heap p h))
  = option_map (permute_value_and_heap p)
      (let* v := eval_prim statics prim vs in Some (v, h)).
Proof.
  rewrite eval_prim_permuted.
  destruct (eval_prim statics prim vs); reflexivity.
Qed.

Lemma phys_equal_on_location_permute (p : permutation)
    (b : binary_primitive) (vs : list value) :
  phys_equal_on_location b (map (permute_value p) vs)
  = phys_equal_on_location b vs.
Proof.
  destruct b; destruct vs as [ | v1 [ | v2 [ | v3 vs]]]; simpl;
    try reflexivity.
  rewrite !is_location_permute. reflexivity.
Qed.

Lemma binary_on_cell_permute (p : permutation) (b : binary_primitive)
    (vs : list value) :
  binary_on_cell b (map (permute_value p) vs)
  = option_map
      (fun read => (permute p (fst read), permute_value p (snd read)))
      (binary_on_cell b vs).
Proof.
  destruct b; destruct vs as [ | v1 [ | v2 [ | v3 vs]]]; try reflexivity;
    destruct v1; reflexivity.
Qed.

Lemma exec_heap_write_permuted (p : permutation) (statics : static_env)
    (h : heap) (hp : heap_primitive) (vs : list value) :
  binds_location (Heap hp) = false ->
  exec_heap_primitive (permute_static_env p statics) (permute_heap p h) hp
    (map (permute_value p) vs)
  = option_map (permute_value_and_heap p)
      (exec_heap_primitive statics h hp vs).
Proof.
  destruct hp as [mut tag | kind init field | kind | akind set_kind];
    intros Hbinds; try discriminate Hbinds.
  - destruct vs as [ | v [ | new [ | v3 vs]]]; try reflexivity.
    change (map (permute_value p) [v; new])
      with [permute_value p v; permute_value p new].
    rewrite !exec_block_set, location_of_permute.
    destruct (location_of v) as [l | ]; simpl; [ | reflexivity ].
    rewrite lookup_permute_heap.
    destruct (lookup l h) as [[[ | ] tag fields | akind elements] | ];
      try reflexivity.
    simpl. rewrite store_field_permute.
    destruct (store_field kind field new tag fields); reflexivity.
  - destruct vs as [ | v [ | index [ | new [ | v4 vs]]]]; try reflexivity.
    change (map (permute_value p) [v; index; new])
      with [permute_value p v; permute_value p index; permute_value p new].
    rewrite !exec_array_set, location_of_permute.
    destruct (location_of v) as [l | ]; simpl; [ | reflexivity ].
    rewrite lookup_permute_heap.
    destruct (lookup l h) as [[mut tag fields | kind elements] | ];
      try reflexivity.
    simpl. rewrite array_stored_permute.
    destruct (array_stored set_kind kind elements index new); reflexivity.
Qed.

Lemma exec_bound_prim_permuted_exactly (p : permutation)
    (statics : static_env) (h : heap) (prim : prim) (vs : list value) :
  binds_location prim = false ->
  exec_bound_prim (permute_static_env p statics) (permute_heap p h) prim
    (map (permute_value p) vs)
  = option_map (permute_value_and_heap p)
      (exec_bound_prim statics h prim vs).
Proof.
  intros Hbinds. destruct prim as [u | b | vp | hp | rp].
  - destruct vs as [ | v [ | v2 vs]].
    + exact (pure_permuted p statics h (Unary u) []).
    + change (map (permute_value p) [v]) with [permute_value p v].
      rewrite !exec_bound_prim_unary, location_of_permute.
      destruct (location_of v) as [l | ]; simpl.
      * rewrite lookup_permute_heap.
        destruct (lookup l h) as [c | ]; simpl; [ | reflexivity ].
        rewrite exec_unary_on_cell_permute.
        destruct (exec_unary_on_cell u l c); reflexivity.
      * exact (pure_permuted p statics h (Unary u) [v]).
    + destruct v;
        match goal with
        | |- context [exec_bound_prim statics h (Unary u) ?l] =>
          exact (pure_permuted p statics h (Unary u) l)
        end.
  - rewrite !exec_bound_prim_binary, phys_equal_on_location_permute,
      binary_on_cell_permute.
    destruct (phys_equal_on_location b vs).
    + destruct b; try reflexivity.
      destruct vs as [ | v1 [ | v2 [ | v3 vs]]]; try reflexivity.
      simpl. rewrite eval_phys_equal_location_permute.
      destruct (eval_phys_equal_location statics v1 v2); reflexivity.
    + destruct (binary_on_cell b vs) as [[l v2] | ]; simpl.
      * rewrite lookup_permute_heap.
        destruct (lookup l h) as [c | ]; simpl; [ | reflexivity ].
        rewrite exec_binary_on_cell_permute.
        destruct (exec_binary_on_cell b l c v2); reflexivity.
      * exact (pure_permuted p statics h (Binary b) vs).
  - exact (pure_permuted p statics h (Variadic vp) vs).
  - exact (exec_heap_write_permuted p statics h hp vs Hbinds).
  - exact (pure_permuted p statics h (Region_primitive rp) vs).
Qed.

Theorem exec_prim_permuted_exactly : exec_prim_permuted_exactly_statement.
Proof.
  intros p statics h prim vs Hbinds. unfold exec_prim.
  rewrite arguments_bound_permute.
  destruct (forallb (argument_bound statics h) vs);
    [ exact (exec_bound_prim_permuted_exactly p statics h prim vs Hbinds)
    | reflexivity ].
Qed.

(* ======================================================================== *)
(* A primitive that binds a location                                        *)
(* ======================================================================== *)

(* The first permutation, then the swap of the fresh location moved and
   the fresh location of the heap moved. *)
Definition after_allocation (p : permutation) (h : heap) : permutation :=
  (permute p (FlambdaHeapSpec.fresh_location h),
   FlambdaHeapSpec.fresh_location (permute_heap p h))
  :: p.

Lemma after_allocation_agrees (p : permutation) (h : heap) (l : location) :
  is_bound l h = true -> permute (after_allocation p h) l = permute p l.
Proof.
  intros Hbound. unfold after_allocation. simpl. apply swap_other.
  - intros Hsame. apply permute_injective in Hsame. subst l.
    unfold is_bound in Hbound. rewrite fresh_location in Hbound.
    discriminate.
  - intros Hsame. pose proof (is_bound_permute_heap p h l) as Hmoved.
    rewrite Hbound, Hsame in Hmoved. unfold is_bound in Hmoved.
    rewrite fresh_location in Hmoved. discriminate.
Qed.

Lemma after_allocation_fresh (p : permutation) (h : heap) :
  permute (after_allocation p h) (FlambdaHeapSpec.fresh_location h)
  = FlambdaHeapSpec.fresh_location (permute_heap p h).
Proof. unfold after_allocation. simpl. apply swap_first. Qed.

Lemma act_after_allocation {T : Type} (act : permutation -> T -> T)
    (names : T -> list positive) (p : permutation) (h : heap) (t : T) :
  action_laws act names ->
  locations_bound h (names t) ->
  act (after_allocation p h) t = act p t.
Proof.
  intros Hlaws Hbound. apply (act_ext _ _ Hlaws). intros name Hname.
  apply after_allocation_agrees, Hbound, Hname.
Qed.

Lemma exec_heap_allocation_permuted (p : permutation) (statics : static_env)
    (h : heap) (hp : heap_primitive) (vs : list value) :
  binds_location (Heap hp) = true ->
  locations_bound h (locations_of_heap h) ->
  locations_bound h (locations_of_values vs) ->
  exec_heap_primitive (permute_static_env p statics) (permute_heap p h) hp
    (map (permute_value p) vs)
  = option_map (permute_value_and_heap (after_allocation p h))
      (exec_heap_primitive statics h hp vs).
Proof.
  intros Hbinds Hheap Hvs.
  pose proof (act_after_allocation _ _ p h h heap_laws Hheap) as Hh.
  pose proof (act_after_allocation _ _ p h vs values_laws Hvs) as Hvs'.
  cbv beta in Hvs'.
  destruct hp as [mut tag | kind init field | kind | akind set_kind];
    simpl in Hbinds; try discriminate; simpl.
  - rewrite (forallb_map_moved _ (fun v => kind_eqb (kind_of_value v) Value))
      by (intros v; rewrite kind_of_permute_value; reflexivity).
    destruct (forallb _ vs); [ | reflexivity ].
    rewrite <- (after_allocation_fresh p h), <- Hh, <- Hvs'. reflexivity.
  - destruct vs as [ | v vs]; [ reflexivity | ].
    change (permute_value p v :: map (permute_value p) vs)
      with (map (permute_value p) (v :: vs)).
    rewrite array_made_permute. unfold array_made.
    destruct (forallb (array_element_ok kind) (v :: vs)); [ | reflexivity ].
    cbn [option_map].
    rewrite <- (after_allocation_fresh p h), <- Hh, <- Hvs'. reflexivity.
Qed.

Theorem exec_prim_permuted : exec_prim_permuted_statement.
Proof.
  intros p statics h prim vs Hheap Hvs.
  destruct (binds_location prim) eqn:Hbinds.
  - exists (after_allocation p h).
    split; [ intros l; apply after_allocation_agrees | ].
    unfold exec_prim. rewrite arguments_bound_permute.
    destruct (forallb (argument_bound statics h) vs); [ | reflexivity ].
    destruct prim as [u | b | vp | hp | rp]; try discriminate.
    exact (exec_heap_allocation_permuted p statics h hp vs Hbinds Hheap Hvs).
  - exists p.
    split;
      [ reflexivity
      | exact (exec_prim_permuted_exactly p statics h prim vs Hbinds) ].
Qed.

(* ======================================================================== *)
(* The premise is kept                                                      *)
(* ======================================================================== *)

Lemma is_bound_head (l : location) (c : cell) (h : heap) :
  is_bound l ((l, c) :: h) = true.
Proof. unfold is_bound. simpl. rewrite Pos.eqb_refl. reflexivity. Qed.

Lemma is_bound_entry (l l0 : location) (c : cell) (h : heap) :
  is_bound l h = true -> is_bound l ((l0, c) :: h) = true.
Proof.
  unfold is_bound. simpl.
  destruct (Pos.eqb l l0); [ reflexivity | intros Hbound; exact Hbound ].
Qed.

Lemma locations_bound_values (h : heap) (vs : list value) (v : value) :
  locations_bound h (locations_of_values vs) ->
  v ∈ vs ->
  locations_bound h (locations_of_value v).
Proof.
  intros Hvs Hv l Hl. apply Hvs. apply in_flat_map. exists v.
  split; assumption.
Qed.

(* The values of a cell that a lookup finds. *)
Lemma locations_bound_lookup (h : heap) (l : location) (c : cell)
    (v : value) :
  locations_bound h (locations_of_heap h) ->
  lookup l h = Some c ->
  v ∈ values_of_cell c ->
  locations_bound h (locations_of_value v).
Proof.
  intros Hheap Hc Hv l0 Hl0. apply Hheap. apply in_flat_map.
  exists (l, c). split; [ exact (lookup_in _ _ _ Hc) | ].
  simpl. right. apply in_flat_map. exists v. split; assumption.
Qed.

(* A heap with one more entry. *)
Lemma locations_bound_entry (h : heap) (l : location) (c : cell) :
  locations_bound h (locations_of_heap h) ->
  (∀ v : value,
     v ∈ values_of_cell c -> locations_bound h (locations_of_value v)) ->
  locations_bound ((l, c) :: h) (locations_of_heap ((l, c) :: h)).
Proof.
  intros Hheap Hcell l0 Hl0. simpl in Hl0.
  destruct Hl0 as [<- | Hl0]; [ apply is_bound_head | ].
  apply is_bound_entry. apply in_app_or in Hl0 as [Hl0 | Hl0].
  - apply in_flat_map in Hl0 as (v & Hv & Hl0). exact (Hcell v Hv l0 Hl0).
  - exact (Hheap l0 Hl0).
Qed.

Lemma eval_prim_locations_bound (statics : static_env) (h : heap)
    (prim : prim) (vs : list value) (v : value) :
  locations_bound h (locations_of_values vs) ->
  locations_bound h (locations_of_static_env statics) ->
  eval_prim statics prim vs = Some v ->
  locations_bound h (locations_of_value v).
Proof.
  intros Hvs Hstatics Heval.
  apply
    (FlambdaPrimitiveProofs.eval_prim_preserves
       (fun w => locations_bound h (locations_of_value w)) statics prim vs v);
    [ | | | | | | exact Heval ].
  - intros c. split; intros l [ ].
  - intros tag fields. split.
    + intros Hblock w Hw l Hl. apply Hblock. simpl. apply in_flat_map.
      exists w. split; assumption.
    + intros Hfields l Hl. simpl in Hl.
      apply in_flat_map in Hl as (w & Hw & Hl). exact (Hfields w Hw l Hl).
  - intros elements. split.
    + intros Harray w Hw l Hl. apply Harray. simpl. apply in_flat_map.
      exists w. split; assumption.
    + intros Helements l Hl. simpl in Hl.
      apply in_flat_map in Hl as (w & Hw & Hl).
      exact (Helements w Hw l Hl).
  - intros s tag fields _ Hs w Hw l Hl. apply Hstatics. apply in_flat_map.
    exists (s, Static_block tag fields).
    split; [ exact (lookup_in _ _ _ Hs) | ].
    simpl. apply in_flat_map. exists w. split; assumption.
  - intros s elements _ Hs w Hw l Hl. apply Hstatics. apply in_flat_map.
    exists (s, Static_immutable_array elements).
    split; [ exact (lookup_in _ _ _ Hs) | ].
    simpl. apply in_flat_map. exists w. split; assumption.
  - intros arg Harg. exact (locations_bound_values h vs arg Hvs Harg).
Qed.

Theorem exec_prim_keeps_locations_bound :
  exec_prim_keeps_locations_bound_statement.
Proof.
  intros statics h h' prim vs v Hheap Hvs Hstatics Hexec.
  apply exec_prim_bound in Hexec as [_ Hexec]. revert Hexec.
  assert (Hpure :
            (let* v0 := eval_prim statics prim vs in Some (v0, h))
            = Some (v, h') ->
            locations_bound h' (locations_of_heap h')
            ∧ locations_bound h' (locations_of_value v)
            ∧ (∀ l : location, is_bound l h = true -> is_bound l h' = true)).
  { destruct (eval_prim statics prim vs) as [v0 | ] eqn:Heval; simpl;
      [ | discriminate ].
    intros [= <- <-].
    split; [ exact Hheap | split; [ | intros l Hl; exact Hl ] ].
    exact (eval_prim_locations_bound statics h prim vs v0 Hvs Hstatics Heval).
  }
  assert (Hconst :
            ∀ (h0 : heap) (c : const),
              locations_bound h0 (locations_of_value (Value_const c))).
  { intros h0 c l [ ]. }
  destruct prim as [u | b | vp | hp | rp]; try exact Hpure.
  - destruct vs as [ | v0 [ | v2 vs]]; try exact Hpure.
    2: { destruct v0; exact Hpure. }
    rewrite exec_bound_prim_unary.
    destruct (location_of v0) as [l | ] eqn:Hlocation.
    2: { destruct v0; try discriminate; exact Hpure. }
    destruct v0; try discriminate. injection Hlocation as ->.
    destruct (lookup l h) as [c | ] eqn:Hc; simpl; [ | discriminate ].
    destruct (exec_unary_on_cell u l c) as [res | ] eqn:Hres; simpl;
      [ | discriminate ].
    intros [= <- <-].
    split; [ exact Hheap | split; [ | intros l0 Hl0; exact Hl0 ] ].
    assert (Hself : locations_bound h (locations_of_value (Value_location l))).
    { intros l0 [<- | [ ]]. unfold is_bound. rewrite Hc. reflexivity. }
    destruct c as [cell_mut tag fields | akind elements];
      destruct u; simpl in Hres; try discriminate.
    all:
      first
        [ injection Hres as <-; apply Hconst
        | destruct (load_allowed _ cell_mut); [ | discriminate ];
          apply load_field_in in Hres;
          exact (locations_bound_lookup h l _ res Hheap Hc Hres)
        | destruct (kind_eqb _ Value); [ | discriminate ];
          injection Hres as <-; exact Hself
        | unfold array_length_of in Hres;
          destruct (in_range _ _); [ | discriminate ];
          injection Hres as <-; apply Hconst
        | match type of Hres with
          | (if ?b then _ else _) = _ => destruct b
          end;
          try discriminate; injection Hres as <-; apply Hconst ].
  - rewrite exec_bound_prim_binary.
    destruct (phys_equal_on_location b vs).
    + destruct b; try discriminate.
      destruct vs as [ | v1 [ | v2 [ | v3 vs]]]; try discriminate.
      destruct (eval_phys_equal_location statics v1 v2); simpl;
        [ | discriminate ].
      intros [= <- <-].
      split; [ exact Hheap | ].
      split; [ apply Hconst | intros l0 Hl0; exact Hl0 ].
    + destruct (binary_on_cell b vs) as [[l v2] | ]; [ | exact Hpure ].
      destruct (lookup l h) as [c | ] eqn:Hc; simpl; [ | discriminate ].
      destruct (exec_binary_on_cell b l c v2) as [res | ] eqn:Hres; simpl;
        [ | discriminate ].
      intros [= <- <-].
      split; [ exact Hheap | split; [ | intros l0 Hl0; exact Hl0 ] ].
      destruct c as [cell_mut tag fields | akind elements];
        [ destruct b; simpl in Hres; discriminate | ].
      destruct b; simpl in Hres; try discriminate.
      destruct (load_allowed _ Cell_mutable); [ | discriminate ].
      apply FlambdaPrimitiveProofs.array_loaded_element in Hres as [Hin _].
      exact (locations_bound_lookup h l _ res Hheap Hc Hin).
  - change (exec_bound_prim statics h (Heap hp) vs)
      with (exec_heap_primitive statics h hp vs).
    destruct hp as [mut tag | kind init field | kind | akind set_kind].
    + simpl. destruct (forallb _ vs); [ | discriminate ].
      intros [= <- <-]. split; [ | split ].
      * apply locations_bound_entry; [ exact Hheap | ].
        intros w Hw. exact (locations_bound_values h vs w Hvs Hw).
      * intros l0 [<- | [ ]]. apply is_bound_head.
      * intros l0. apply is_bound_entry.
    + destruct vs as [ | v0 [ | new [ | v3 vs]]]; try (simpl; discriminate).
      rewrite exec_block_set.
      destruct (location_of v0) as [l | ]; [ | discriminate ].
      destruct (lookup l h) as [[[ | ] tag fields | akind elements] | ]
        eqn:Hc;
        try discriminate.
      destruct (store_field kind field new tag fields) as [fields1 | ]
        eqn:Hstore; simpl; [ | discriminate ].
      intros [= <- <-].
      destruct (store_field_in _ _ _ _ _ _ Hstore) as [_ Hin].
      split; [ | split ].
      * apply locations_bound_entry; [ exact Hheap | ].
        intros w Hw. simpl in Hw. destruct (Hin w Hw) as [-> | Hold].
        -- apply (locations_bound_values h [v0; new] new Hvs).
           right. left. reflexivity.
        -- exact (locations_bound_lookup h l _ w Hheap Hc Hold).
      * apply Hconst.
      * intros l0. apply is_bound_entry.
    + simpl. destruct vs as [ | v0 vs0]; [ discriminate | ].
      destruct (array_made kind (v0 :: vs0)) as [elements | ] eqn:Hmade;
        simpl; [ | discriminate ].
      intros [= <- <-].
      apply FlambdaPrimitiveProofs.array_made_of_arguments in Hmade
        as [-> _].
      split; [ | split ].
      * apply locations_bound_entry; [ exact Hheap | ].
        intros w Hw. exact (locations_bound_values h _ w Hvs Hw).
      * intros l0 [<- | [ ]]. apply is_bound_head.
      * intros l0. apply is_bound_entry.
    + destruct vs as [ | v0 [ | index [ | new [ | v4 vs]]]];
        try (simpl; discriminate).
      rewrite exec_array_set.
      destruct (location_of v0) as [l | ]; [ | discriminate ].
      destruct (lookup l h) as [[mut tag fields | kind elements] | ] eqn:Hc;
        try discriminate.
      destruct (array_stored set_kind kind elements index new)
        as [stored | ] eqn:Hstore; simpl; [ | discriminate ].
      intros [= <- <-].
      destruct
        (FlambdaPrimitiveProofs.array_stored_elements _ _ _ _ _ _ Hstore)
        as (_ & _ & Hin).
      split; [ | split ].
      * apply locations_bound_entry; [ exact Hheap | ].
        intros w Hw. simpl in Hw. destruct (Hin w Hw) as [-> | Hold].
        -- apply (locations_bound_values h [v0; index; new] new Hvs).
           right. right. left. reflexivity.
        -- exact (locations_bound_lookup h l _ w Hheap Hc Hold).
      * apply Hconst.
      * intros l0. apply is_bound_entry.
Qed.

(* ======================================================================== *)
(* The premise counts the entries that a write hides                        *)
(* ======================================================================== *)

Theorem hidden_entry : hidden_entry_statement.
Proof.
  split.
  - split.
    + intros l c v Hl Hv. unfold hidden_entry_heap in Hl.
      destruct l; simpl in Hl; try discriminate.
      injection Hl as <-. destruct Hv.
    + intros s a v Hs. simpl in Hs. discriminate Hs.
  - intros [p' Hsame]. vm_compute in Hsame.
    injection Hsame as Hkey _ Hfield. congruence.
Qed.

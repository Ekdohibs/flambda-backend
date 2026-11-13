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

(* Proofs of the statements of FlambdaHeapSpec.v. No review needed. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec FlambdaMapProofs
  FlambdaValueSpec FlambdaValueProofs FlambdaPrimitiveModel
  FlambdaPrimitiveSpec FlambdaHeapSpec.
From OxCamlFlambda2 Require FlambdaPrimitiveProofs.
Import ListNotations.

(* FlambdaPrimitiveProofs.v makes [eval_prim] opaque for the files that
   require it. The proofs here go by cases on what a primitive computes,
   so they need it to compute. *)
#[local] Transparent eval_prim.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* Allocation                                                               *)
(* ======================================================================== *)

(* [fresh_location] is fresh: it is greater than every location that the
   heap binds. *)
Lemma max_bound (l : positive) (names : list positive) :
  l ∈ names -> (l <= fold_right Pos.max 1 names)%positive.
Proof.
  induction names as [ | name names IH]; simpl; [ contradiction | ].
  intros [-> | Hin]; [ lia | ]. specialize (IH Hin). lia.
Qed.

Theorem fresh_location : fresh_location_statement.
Proof.
  intros h.
  unfold FlambdaHeapSpec.fresh_location.
  destruct (lookup _ h) as [c | ] eqn:Hlookup; [ | reflexivity ].
  exfalso.
  assert (Hbound :
            is_bound (Pos.succ (fold_right Pos.max 1%positive (domain h))) h
            = true).
  { unfold is_bound. rewrite Hlookup. reflexivity. }
  rewrite <- among_domain in Hbound. apply among_In in Hbound.
  apply max_bound in Hbound. lia.
Qed.

Lemma lookup_fresh_location (l : location) (c added : cell) (h : heap) :
  lookup l h = Some c ->
  lookup l ((FlambdaHeapSpec.fresh_location h, added) :: h) = Some c.
Proof.
  intros Hlookup. simpl.
  destruct (Pos.eqb_spec l (FlambdaHeapSpec.fresh_location h)) as [-> | _];
    [ | exact Hlookup ].
  rewrite fresh_location in Hlookup. discriminate.
Qed.

(* ======================================================================== *)
(* The laws of effects and coeffects                                        *)
(* ======================================================================== *)

Lemma heap_evolves_refl (h : heap) : heap_evolves h h.
Proof.
  split.
  - intros l mut tag fields Hlookup. exists fields. auto.
  - intros l kind elements Hlookup. exists elements. auto.
Qed.

Lemma heap_evolves_trans (h1 h2 h3 : heap) :
  heap_evolves h1 h2 -> heap_evolves h2 h3 -> heap_evolves h1 h3.
Proof.
  intros [H12 A12] [H23 A23]. split.
  - intros l mut tag fields Hlookup.
    destruct (H12 _ _ _ _ Hlookup)
      as (fields2 & Hlookup2 & Hlength2 & Hsame2).
    destruct (H23 _ _ _ _ Hlookup2)
      as (fields3 & Hlookup3 & Hlength3 & Hsame3).
    exists fields3. split; [ exact Hlookup3 | ]. split; [ congruence | ].
    intros Hunique. rewrite (Hsame3 Hunique). exact (Hsame2 Hunique).
  - intros l kind elements Hlookup.
    destruct (A12 _ _ _ Hlookup) as (elements2 & Hlookup2 & Hlength2).
    destruct (A23 _ _ _ Hlookup2) as (elements3 & Hlookup3 & Hlength3).
    exists elements3. split; [ exact Hlookup3 | congruence ].
Qed.

(* A location that is bound stays bound as the heap evolves. *)
Lemma heap_evolves_bound (h h' : heap) (l : location) :
  heap_evolves h h' -> is_bound l h = true -> is_bound l h' = true.
Proof.
  intros [Hblocks Harrays]. unfold is_bound.
  destruct (lookup l h) as [[mut tag fields | kind elements] | ] eqn:Hl;
    [ | | discriminate ].
  - destruct (Hblocks _ _ _ _ Hl) as (fields' & -> & _). reflexivity.
  - destruct (Harrays _ _ _ Hl) as (elements' & -> & _). reflexivity.
Qed.

(* A heap evolves to itself with a cell at a fresh location, to itself
   with the fields of a mutable block replaced by as many, and to
   itself with the elements of an array replaced by as many. *)
Lemma heap_evolves_fresh (h : heap) (c : cell) :
  heap_evolves h ((FlambdaHeapSpec.fresh_location h, c) :: h).
Proof.
  split.
  - intros l mut tag fields Hlookup. exists fields.
    split; [ apply lookup_fresh_location; exact Hlookup | auto ].
  - intros l kind elements Hlookup. exists elements.
    split; [ apply lookup_fresh_location; exact Hlookup | reflexivity ].
Qed.

Lemma heap_evolves_write_block (h : heap) (l : location) (tag : Z)
    (fields fields' : list value) :
  lookup l h = Some (Cell_block Cell_mutable tag fields) ->
  length fields' = length fields ->
  heap_evolves h ((l, Cell_block Cell_mutable tag fields') :: h).
Proof.
  intros Hcell Hlength. split.
  - intros l' mut' tag' fields0 Hlookup. simpl.
    destruct (Pos.eqb_spec l' l) as [-> | _].
    + rewrite Hcell in Hlookup. injection Hlookup as <- <- <-.
      exists fields'. split; [ reflexivity | ].
      split; [ exact Hlength | discriminate ].
    + exists fields0. auto.
  - intros l' kind elements Hlookup. simpl.
    destruct (Pos.eqb_spec l' l) as [-> | _].
    + rewrite Hcell in Hlookup. discriminate Hlookup.
    + exists elements. auto.
Qed.

Lemma heap_evolves_write_array (h : heap) (l : location) (kind : array_kind)
    (elements elements' : list value) :
  lookup l h = Some (Cell_array kind elements) ->
  length elements' = length elements ->
  heap_evolves h ((l, Cell_array kind elements') :: h).
Proof.
  intros Hcell Hlength. split.
  - intros l' mut' tag' fields Hlookup. simpl.
    destruct (Pos.eqb_spec l' l) as [-> | _].
    + rewrite Hcell in Hlookup. discriminate Hlookup.
    + exists fields. auto.
  - intros l' kind' elements0 Hlookup. simpl.
    destruct (Pos.eqb_spec l' l) as [-> | _].
    + rewrite Hcell in Hlookup. injection Hlookup as <- <-.
      exists elements'. split; [ reflexivity | exact Hlength ].
    + exists elements0. auto.
Qed.

Lemma store_field_length (kind : block_access_kind) (field : Z) (new : value)
    (tag : Z) (fields fields' : list value) :
  store_field kind field new tag fields = Some fields' ->
  length fields' = length fields.
Proof.
  destruct kind as [known_tag size field_kind]. simpl.
  destruct (_ && _); [ | discriminate ].
  apply FlambdaPrimitiveProofs.replace_nth_length.
Qed.

Lemma exec_heap_primitive_evolves (statics : static_env) (h h' : heap)
    (p : heap_primitive) (vs : list value) (v : value) :
  exec_heap_primitive statics h p vs = Some (v, h') -> heap_evolves h h'.
Proof.
  destruct p as [mut tag | kind init field | kind | kind set_kind]; simpl.
  - destruct (forallb _ vs); [ | discriminate ]. intros [= <- <-].
    apply heap_evolves_fresh.
  - destruct vs as [ | v1 [ | v2 [ | v3 vs]]]; try discriminate.
    destruct v1 as [c | code slots | fslot decls slots | s | tag fields | | c
      | l | callee0 applied applied_arity];
      try discriminate.
    destruct (lookup l h) as [[[ | ] tag fields | akind elements] | ]
      eqn:Hcell;
      try discriminate.
    destruct (store_field kind field v2 tag fields) as [fields' | ] eqn:Hstore;
      simpl; [ | discriminate ].
    intros [= <- <-].
    exact
      (heap_evolves_write_block _ _ _ _ _ Hcell
         (store_field_length _ _ _ _ _ _ Hstore)).
  - (* [Make_mutable_array] *)
    destruct vs as [ | v0 vs0]; [ discriminate | ].
    destruct (array_made kind (v0 :: vs0)) as [elements | ]; simpl;
      [ | discriminate ].
    intros [= <- <-]. apply heap_evolves_fresh.
  - (* [Array_set] *)
    destruct vs as [ | v1 [ | v2 [ | v3 [ | v4 vs]]]]; try discriminate.
    destruct v1 as [c | code slots | fslot decls slots | s | tag fields | | c
      | l | callee0 applied applied_arity];
      try discriminate.
    destruct (lookup l h) as [[mut tag fields | akind elements] | ] eqn:Hcell;
      try discriminate.
    destruct (array_stored set_kind akind elements v2 v3) as [elements' | ]
      eqn:Hstore;
      simpl; [ | discriminate ].
    intros [= <- <-].
    exact
      (heap_evolves_write_array _ _ _ _ _ Hcell
         (proj1 (FlambdaPrimitiveProofs.array_stored_elements _ _ _ _ _ _
                   Hstore))).
Qed.

(* A primitive that has a value was given arguments that are bound. *)
Lemma exec_prim_bound (statics : static_env) (h h' : heap) (p : prim)
    (vs : list value) (v : value) :
  exec_prim statics h p vs = Some (v, h') ->
  forallb (argument_bound statics h) vs = true
  ∧ exec_bound_prim statics h p vs = Some (v, h').
Proof.
  unfold exec_prim. destruct (forallb _ vs); [ auto | discriminate ].
Qed.

Theorem exec_prim_symbols_bound : exec_prim_symbols_bound_statement.
Proof.
  intros statics h h' p vs v s Hexec Hs.
  apply exec_prim_bound in Hexec as [Hbound _].
  rewrite forallb_forall in Hbound. specialize (Hbound _ Hs).
  simpl in Hbound. unfold is_bound in Hbound.
  destruct (lookup s (static_values statics));
    [ discriminate | discriminate Hbound ].
Qed.

Theorem exec_prim_locations_bound : exec_prim_locations_bound_statement.
Proof.
  intros statics h h' p vs v l Hexec Hl.
  apply exec_prim_bound in Hexec as [Hbound _].
  rewrite forallb_forall in Hbound. specialize (Hbound _ Hl).
  simpl in Hbound. unfold is_bound in Hbound.
  destruct (lookup l h); [ discriminate | discriminate Hbound ].
Qed.

(* Arguments that are bound stay bound as the heap evolves. *)
Lemma arguments_bound_evolves (statics : static_env) (h1 h2 : heap)
    (vs : list value) :
  heap_evolves h1 h2 ->
  forallb (argument_bound statics h1) vs = true ->
  forallb (argument_bound statics h2) vs = true.
Proof.
  intros Hevolves. rewrite !forallb_forall. intros Hall v Hv.
  specialize (Hall v Hv). destruct v; try exact Hall.
  exact (heap_evolves_bound _ _ _ Hevolves Hall).
Qed.

(* [Phys_equal] with a location among its two arguments: the one
   application of a binary primitive that is not
   FlambdaPrimitiveSpec.eval_prim. *)
Definition phys_equal_on_location (p : binary_primitive) (vs : list value)
    : bool :=
  match p, vs with
  | Phys_equal _, [v1; v2] => is_location v1 || is_location v2
  | _, _ => false
  end.

(* The location and the other argument of a binary primitive that reads
   a cell: its first argument is a location, and it is not
   [Phys_equal]. *)
Definition binary_on_cell (p : binary_primitive) (vs : list value)
    : option (location * value) :=
  match p, vs with
  | Phys_equal _, _ => None
  | _, [Value_location l; v2] => Some (l, v2)
  | _, _ => None
  end.

Lemma exec_bound_prim_binary (statics : static_env) (h : heap)
    (p : binary_primitive) (vs : list value) :
  exec_bound_prim statics h (Binary p) vs
  = if phys_equal_on_location p vs
    then
      match p, vs with
      | Phys_equal op, [v1; v2] =>
        let* same := eval_phys_equal_location statics v1 v2 in
        Some
          (naked_bool
             match op with
             | Phys_eq => same
             | Phys_neq => negb same
             end,
           h)
      | _, _ => None
      end
    else
      match binary_on_cell p vs with
      | Some (l, v2) =>
        let* c := lookup l h in
        let* v := exec_binary_on_cell p l c v2 in
        Some (v, h)
      | None =>
        let* v := eval_prim statics (Binary p) vs in
        Some (v, h)
      end.
Proof.
  destruct p;
    destruct vs as [ | v1 [ | v2 [ | v3 vs]]]; try reflexivity;
    destruct v1; try reflexivity;
    simpl; destruct (is_location v2); reflexivity.
Qed.

(* A primitive that is not of the heap returns the heap as it came. *)
Lemma exec_bound_prim_not_heap (statics : static_env) (h h' : heap) (p : prim)
    (vs : list value) (v : value) :
  (∀ p' : heap_primitive, p <> Heap p') ->
  exec_bound_prim statics h p vs = Some (v, h') ->
  h' = h.
Proof.
  intros Hnot_heap.
  assert (Hpure :
            (let* v := eval_prim statics p vs in Some (v, h)) = Some (v, h')
            -> h' = h).
  { destruct (eval_prim statics p vs); simpl; [ | discriminate ].
    intros [= _ <-]. reflexivity. }
  assert (Hcell :
            ∀ (p' : unary_primitive) (l : location),
              (let* c := lookup l h in
               let* v := exec_unary_on_cell p' l c in
               Some (v, h))
              = Some (v, h')
              -> h' = h).
  { intros p' l. destruct (lookup l h) as [c | ]; simpl; [ | discriminate ].
    destruct (exec_unary_on_cell p' l c); simpl; [ | discriminate ].
    intros [= _ <-]. reflexivity. }
  destruct p as [p | p | p | p | rp].
  - destruct vs as [ | v1 [ | v2 vs]]; try exact Hpure;
      destruct v1; try exact Hpure.
    apply Hcell.
  - rewrite exec_bound_prim_binary.
    destruct (phys_equal_on_location p vs).
    + destruct p; try discriminate.
      destruct vs as [ | v1 [ | v2 [ | v3 vs]]]; try discriminate.
      destruct (eval_phys_equal_location statics v1 v2); simpl;
        [ | discriminate ].
      intros [= _ <-]. reflexivity.
    + destruct (binary_on_cell p vs) as [[l v2] | ]; [ | exact Hpure ].
      destruct (lookup l h) as [c | ]; simpl; [ | discriminate ].
      destruct (exec_binary_on_cell p l c v2); simpl; [ | discriminate ].
      intros [= _ <-]. reflexivity.
  - exact Hpure.
  - exfalso. exact (Hnot_heap p eq_refl).
  - exact Hpure.
Qed.

Theorem exec_prim_evolves : exec_prim_evolves_statement.
Proof.
  intros statics h h' p vs v Hexec.
  apply exec_prim_bound in Hexec as [_ Hexec].
  destruct p as [p | p | p | p | rp].
  4: { exact (exec_heap_primitive_evolves _ _ _ _ _ _ Hexec). }
  all: apply exec_bound_prim_not_heap in Hexec;
    [ subst h'; apply heap_evolves_refl
    | intros p' Hheap; discriminate Hheap ].
Qed.

Theorem exec_prim_arrays : exec_prim_arrays_statement.
Proof.
  intros statics h h' p vs v Harrays Hexec.
  apply exec_prim_bound in Hexec as [_ Hexec].
  destruct p as [p | p | p | p | rp].
  4: {
    change (exec_bound_prim statics h (Heap p) vs)
      with (exec_heap_primitive statics h p vs) in Hexec.
    destruct p as [mut tag | kind init field | kind | akind set_kind];
      simpl in Hexec.
    - destruct (forallb _ vs); [ | discriminate ]. injection Hexec as _ <-.
      intros l kind elements w Hl Hw. simpl in Hl.
      destruct (Pos.eqb l _);
        [ discriminate | exact (Harrays _ _ _ _ Hl Hw) ].
    - destruct vs as [ | v1 [ | v2 [ | v3 vs]]]; try discriminate.
      destruct v1 as [c | code slots | fslot decls slots | s | tag fields | | c
        | l | callee0 applied applied_arity];
        try discriminate.
      destruct (lookup l h) as [[[ | ] tag fields | akind elements] | ];
        try discriminate.
      destruct (store_field kind field v2 tag fields) as [fields' | ];
        simpl in Hexec; [ | discriminate ].
      injection Hexec as _ <-.
      intros l0 kind0 elements w Hl Hw. simpl in Hl.
      destruct (Pos.eqb l0 l);
        [ discriminate | exact (Harrays _ _ _ _ Hl Hw) ].
    - (* [Make_mutable_array]: the arguments are values that the kind
         holds. *)
      destruct vs as [ | v0 vs0]; [ discriminate | ].
      destruct (array_made kind (v0 :: vs0)) as [elements | ] eqn:Hmade;
        simpl in Hexec; [ | discriminate ].
      injection Hexec as _ <-.
      apply FlambdaPrimitiveProofs.array_made_of_arguments in Hmade
        as [-> Hok].
      intros l kind0 elements w Hl Hw. simpl in Hl.
      destruct (Pos.eqb l _).
      + injection Hl as <- <-. exact (Hok w Hw).
      + exact (Harrays _ _ _ _ Hl Hw).
    - (* [Array_set]: so is the new value. *)
      destruct vs as [ | v1 [ | v2 [ | v3 [ | v4 vs]]]]; try discriminate.
      destruct v1 as [c | code slots | fslot decls slots | s | tag fields | | c
        | l | callee0 applied applied_arity];
        try discriminate.
      destruct (lookup l h) as [[mut tag fields | kind elements] | ] eqn:Hcell;
        try discriminate.
      destruct (array_stored set_kind kind elements v2 v3) as [stored | ]
        eqn:Hstore;
        simpl in Hexec; [ | discriminate ].
      injection Hexec as _ <-.
      destruct
        (FlambdaPrimitiveProofs.array_stored_elements _ _ _ _ _ _ Hstore)
        as (_ & Hok & Hin).
      intros l0 kind0 elements0 w Hl Hw. simpl in Hl.
      destruct (Pos.eqb_spec l0 l) as [-> | _].
      + injection Hl as <- <-.
        destruct (Hin w Hw) as [-> | Hold];
          [ exact Hok | exact (Harrays _ _ _ _ Hcell Hold) ].
      + exact (Harrays _ _ _ _ Hl Hw). }
  all: apply exec_bound_prim_not_heap in Hexec;
    [ subst h'; exact Harrays | intros p' Hheap; discriminate Hheap ].
Qed.

Theorem array_length_known : array_length_known_statement.
Proof.
  intros statics h h1 h2 kind length_kind vs v Hmake Hevolves.
  apply exec_prim_bound in Hmake as [_ Hmake]. simpl in Hmake.
  destruct vs as [ | v0 vs0]; [ discriminate | ].
  destruct (array_made kind (v0 :: vs0)) as [elements | ] eqn:Hmade;
    simpl in Hmake; [ | discriminate ].
  injection Hmake as <- <-.
  apply FlambdaPrimitiveProofs.array_made_of_arguments in Hmade as [-> _].
  destruct
    (proj2 Hevolves (FlambdaHeapSpec.fresh_location h) kind (v0 :: vs0))
    as (elements' & Hcell & Hlength).
  { simpl. rewrite Pos.eqb_refl. reflexivity. }
  unfold exec_prim. simpl. unfold is_bound. rewrite Hcell. simpl.
  rewrite (FlambdaPrimitiveProofs.array_length_of_same_length _ _ Hlength).
  reflexivity.
Qed.

Theorem array_kind_of_access : array_kind_of_access_statement.
Proof.
  intros statics h vs kind kind' length_kind length_kind' load_kind mut
    set_kind.
  unfold exec_prim.
  destruct (forallb (argument_bound statics h) vs); [ | auto ].
  split; [ | split ].
  - destruct vs as [ | v1 [ | v2 vs]]; reflexivity.
  - destruct vs as [ | v1 [ | v2 [ | v3 vs]]]; reflexivity.
  - reflexivity.
Qed.

Theorem no_effects : no_effects_statement.
Proof.
  intros statics h h' p vs v Heffects Hexec.
  apply exec_prim_bound in Hexec as [_ Hexec].
  destruct p as [p | p | p | p | rp].
  4: { destruct p as [[ | ] tag | kind init field | kind | kind set_kind];
       discriminate Heffects. }
  all: apply exec_bound_prim_not_heap in Hexec;
    [ exact Hexec | intros p' Hheap; discriminate Hheap ].
Qed.

Theorem only_generative_effects : only_generative_effects_statement.
Proof.
  intros statics h h' p vs v mut Heffects Hexec.
  apply exec_prim_bound in Hexec as [_ Hexec].
  destruct p as [p | p | p | p | rp].
  4: {
    right.
    destruct p as [cell_mut tag | kind init field | kind | kind set_kind];
      try discriminate.
    - simpl in Hexec. destruct (forallb _ vs); [ | discriminate ].
      injection Hexec as <- <-.
      split.
      + destruct cell_mut; simpl in Heffects; injection Heffects as <-;
          discriminate.
      + split; [ reflexivity | eexists; reflexivity ].
    - (* [Make_mutable_array] *)
      simpl in Hexec. destruct vs as [ | v0 vs0]; [ discriminate | ].
      destruct (array_made kind (v0 :: vs0)) as [elements | ];
        simpl in Hexec; [ | discriminate ].
      injection Hexec as <- <-.
      split; [ simpl in Heffects; injection Heffects as <-; discriminate | ].
      split; [ reflexivity | eexists; reflexivity ]. }
  4: { destruct vs; discriminate Hexec. }
  all: left; split;
    [ | apply exec_bound_prim_not_heap in Hexec;
        [ exact Hexec | intros p' Hheap; discriminate Hheap ] ].
  - destruct p; simpl in Heffects; try discriminate.
    injection Heffects as <-. reflexivity.
  - destruct p; discriminate.
  - destruct p; simpl in Heffects; injection Heffects as <-; reflexivity.
Qed.

Theorem no_coeffects : no_coeffects_statement.
Proof.
  intros statics h1 h1' h2 p vs v Heffects Hcoeffects Hevolves Hexec.
  apply exec_prim_bound in Hexec as [Hbound Hexec].
  unfold exec_prim. rewrite (arguments_bound_evolves _ _ _ _ Hevolves Hbound).
  assert (Hpure :
            (let* v := eval_prim statics p vs in Some (v, h1)) = Some (v, h1')
            -> (let* v := eval_prim statics p vs in Some (v, h2))
               = Some (v, h2)).
  { destruct (eval_prim statics p vs); simpl; [ | discriminate ].
    intros [= <- _]. reflexivity. }
  destruct p as [p | p | p | p | rp].
  - destruct vs as [ | v1 [ | v2 vs]]; try exact (Hpure Hexec);
      destruct v1 as [c | code slots | fslot decls slots | s | tag fields
      | elements | c | l | callee0 applied applied_arity];
      try exact (Hpure Hexec).
    simpl in Hexec |- *.
    destruct (lookup l h1) as [[cell_mut tag fields | akind elements] | ]
      eqn:Hcell;
      simpl in Hexec; [ | | discriminate ].
    + destruct (proj1 Hevolves _ _ _ _ Hcell) as (fields' & -> & _ & Hsame).
      simpl.
      destruct p; simpl in Hexec |- *; try discriminate;
        try (injection Hexec as <- _; reflexivity).
      (* The load: marked [Mutable] it has a coeffect; otherwise its cell
         is not mutable, and keeps its fields. *)
      destruct (load_allowed mut cell_mut) eqn:Hallowed; [ | discriminate ].
      assert (Hunique : cell_mut = Cell_immutable_unique).
      { destruct mut, cell_mut; try reflexivity; discriminate. }
      rewrite (Hsame Hunique).
      destruct (load_field kind field tag fields); simpl in Hexec |- *;
        [ | discriminate ].
      injection Hexec as <- _. reflexivity.
    + (* An array: its length is kept. *)
      destruct (proj2 Hevolves _ _ _ Hcell) as (elements' & -> & Hlength).
      simpl.
      destruct p as [ | [ | ] | | | length_kind | | | | | | | | | | ];
        simpl in Hexec |- *; try discriminate;
        try (injection Hexec as <- _; reflexivity).
      rewrite
        (FlambdaPrimitiveProofs.array_length_of_same_length _ _ Hlength).
      destruct (array_length_of elements); simpl in Hexec |- *;
        [ | discriminate ].
      injection Hexec as <- _. reflexivity.
  - rewrite exec_bound_prim_binary in Hexec |- *.
    destruct (phys_equal_on_location p vs).
    + destruct p; try discriminate.
      destruct vs as [ | v1 [ | v2 [ | v3 vs]]]; try discriminate.
      destruct (eval_phys_equal_location statics v1 v2);
        simpl in Hexec |- *; [ | discriminate ].
      injection Hexec as <- _. reflexivity.
    + destruct (binary_on_cell p vs) as [[l v2] | ];
        [ | exact (Hpure Hexec) ].
      (* The load of an array: marked [Mutable] it has a coeffect, and
         otherwise it has no value on an array that has an identity. *)
      destruct (lookup l h1) as [[cell_mut tag fields | akind elements] | ];
        simpl in Hexec; try discriminate.
      destruct p as [array_kind load_kind [ | | ] | | | | | ];
        simpl in Hexec; discriminate.
  - exact (Hpure Hexec).
  - destruct p as [[ | ] tag | kind init field | kind | kind set_kind];
      discriminate Heffects.
  - destruct rp; discriminate Heffects.
Qed.

Theorem mutable_load_not_stable : mutable_load_not_stable_statement.
Proof.
  pose (l := 1%positive).
  pose (one := Value_const (Tagged_immediate 1)).
  pose (two := Value_const (Tagged_immediate 2)).
  pose (h1 := [(l, Cell_block Cell_mutable 0 [one])]).
  pose (h2 := (l, Cell_block Cell_mutable 0 [two]) :: h1).
  exists empty_static_env, h1, h2,
    (Unary (Block_load (Block_access_values None None Any_value) Mutable 0)),
    [Value_location l], one, two.
  split; [ reflexivity | ].
  split.
  { split.
    - intros l' mut tag fields. unfold h1, h2, l.
      destruct l'; simpl; try discriminate.
      intros [= <- <- <-]. exists [two]. split; [ reflexivity | ].
      split; [ reflexivity | discriminate ].
    - intros l' kind elements. unfold h1, l.
      destruct l'; simpl; discriminate. }
  split; [ reflexivity | ]. split; [ reflexivity | discriminate ].
Qed.

(* The primitives of arrays, run on a heap that has an array of two
   integers and a block: so that no statement about them holds for
   their having no value at all, and as a check of the definitions
   against known answers. *)
Module Array_examples.

Definition imm (i : Z) : value := Value_const (Tagged_immediate i).
Definition an_array : location := 1%positive.
Definition a_block : location := 2%positive.
Definition h : heap :=
  [(an_array, Cell_array Array_immediates [imm 5; imm 6]);
   (a_block, Cell_block Cell_mutable 0 [imm 7])].
Definition array : value := Value_location an_array.
Definition block : value := Value_location a_block.

Definition exec (p : prim) (vs : list value) : option (value * heap) :=
  exec_prim empty_static_env h p vs.

Definition load (mut : mutability) : prim :=
  Binary (Array_load Array_immediates Load_immediates mut).

Example make :
  exec_prim empty_static_env empty_heap
    (Heap (Make_mutable_array Array_immediates)) [imm 5; imm 6]
  = Some
      (Value_location (FlambdaHeapSpec.fresh_location empty_heap),
       [(FlambdaHeapSpec.fresh_location empty_heap,
         Cell_array Array_immediates [imm 5; imm 6])]).
Proof. reflexivity. Qed.

Example make_of_no_element :
  exec (Heap (Make_mutable_array Array_values)) [] = None.
Proof. reflexivity. Qed.

Example make_immediates_of_a_location :
  exec (Heap (Make_mutable_array Array_immediates)) [imm 5; array] = None.
Proof. reflexivity. Qed.

Example make_values_of_a_location :
  option_map fst (exec (Heap (Make_mutable_array Array_values)) [imm 5; array])
  = Some (Value_location (FlambdaHeapSpec.fresh_location h)).
Proof. reflexivity. Qed.

Example length :
  exec (Unary (Array_length (Array_kind Array_immediates))) [array]
  = Some (imm 2, h).
Proof. reflexivity. Qed.

Example length_at_another_kind :
  exec (Unary (Array_length Float_array_opt_dynamic)) [array]
  = Some (imm 2, h).
Proof. reflexivity. Qed.

Example length_of_a_block :
  exec (Unary (Array_length (Array_kind Array_values))) [block] = None.
Proof. reflexivity. Qed.

Example load_second : exec (load Mutable) [array; imm 1] = Some (imm 6, h).
Proof. reflexivity. Qed.

Example load_past_the_end : exec (load Mutable) [array; imm 2] = None.
Proof. reflexivity. Qed.

Example load_before_the_start : exec (load Mutable) [array; imm (-1)] = None.
Proof. reflexivity. Qed.

Example load_marked_immutable : exec (load Immutable) [array; imm 1] = None.
Proof. reflexivity. Qed.

Example load_of_a_block : exec (load Mutable) [block; imm 0] = None.
Proof. reflexivity. Qed.

(* The heap after the second element is replaced by 9. *)
Definition h_after_set : heap :=
  (an_array, Cell_array Array_immediates [imm 5; imm 9]) :: h.

Example set :
  exec (Heap (Array_set Array_gc_ignorable_values Set_gc_ignorable_values))
    [array; imm 1; imm 9]
  = Some (unit_value, h_after_set).
Proof. reflexivity. Qed.

Example set_then_load :
  exec_prim empty_static_env h_after_set (load Mutable) [array; imm 1]
  = Some (imm 9, h_after_set).
Proof. reflexivity. Qed.

Example set_past_the_end :
  exec (Heap (Array_set Array_immediates Set_immediates))
    [array; imm 2; imm 9]
  = None.
Proof. reflexivity. Qed.

(* A store at [Values] of what an array made at [Immediates] does not
   hold. *)
Example set_a_location_in_immediates :
  exec (Heap (Array_set Array_values (Set_values Assignment)))
    [array; imm 1; block]
  = None.
Proof. reflexivity. Qed.

Example set_of_a_block :
  exec (Heap (Array_set Array_values (Set_values Assignment)))
    [block; imm 0; imm 9]
  = None.
Proof. reflexivity. Qed.

Example is_int_of_an_array :
  exec (Unary (Is_int false)) [array] = Some (naked_bool false, h).
Proof. reflexivity. Qed.

Example is_int_variant_only_of_an_array :
  exec (Unary (Is_int true)) [array] = None.
Proof. reflexivity. Qed.

Example get_tag_of_an_array : exec (Unary Get_tag) [array] = None.
Proof. reflexivity. Qed.

Example block_load_of_an_array :
  exec (Unary (Block_load (Block_access_values None None Any_value) Mutable 0))
    [array]
  = None.
Proof. reflexivity. Qed.

Example block_set_of_an_array :
  exec
    (Heap
       (Block_set (Block_access_values None None Any_value) Assignment 0))
    [array; imm 9]
  = None.
Proof. reflexivity. Qed.

Example phys_equal_of_an_array_and_itself :
  exec (Binary (Phys_equal Phys_eq)) [array; array]
  = Some (naked_bool true, h).
Proof. reflexivity. Qed.

End Array_examples.

(* ======================================================================== *)
(* Closed memories                                                          *)
(* ======================================================================== *)

(* Induction on a value, through the lists it holds. *)
Lemma value_induction (P : value -> Prop) :
  (∀ c : const, P (Value_const c)) ->
  (∀ (code : code_id) (slots : list (value_slot * value)),
    Forall (fun slot => P (snd slot)) slots -> P (Value_closure code slots)) ->
  (∀ (slot : function_slot) (decls : list (function_slot * code_id))
     (slots : list (value_slot * value)),
    Forall (fun slot => P (snd slot)) slots ->
    P (Value_closure_of_set slot decls slots)) ->
  (∀ s : symbol, P (Value_symbol s)) ->
  (∀ (tag : Z) (fields : list value),
    Forall P fields -> P (Value_block tag fields)) ->
  (∀ elements : list value,
    Forall P elements -> P (Value_immutable_array elements)) ->
  (∀ c : const, P (Value_boxed_number c)) ->
  (∀ l : location, P (Value_location l)) ->
  (∀ (callee : value) (args : list value) (args_arity : arity),
    P callee ->
    Forall P args ->
    P (Value_partial_application callee args args_arity)) ->
  ∀ v : value, P v.
Proof.
  intros Hconst Hclosure Hclosure_of_set Hsymbol Hblock Harray Hboxed
    Hlocation Hpartial.
  fix IH 1.
  intros [c | code slots | slot decls slots | s | tag fields | elements | c
         | l | callee args args_arity].
  - apply Hconst.
  - apply Hclosure.
    exact
      ((fix go (slots : list (value_slot * value))
          : Forall (fun slot => P (snd slot)) slots :=
          match slots with
          | [] => Forall_nil _
          | slot :: slots => Forall_cons _ (IH (snd slot)) (go slots)
          end)
         slots).
  - apply Hclosure_of_set.
    exact
      ((fix go (slots : list (value_slot * value))
          : Forall (fun slot => P (snd slot)) slots :=
          match slots with
          | [] => Forall_nil _
          | slot :: slots => Forall_cons _ (IH (snd slot)) (go slots)
          end)
         slots).
  - apply Hsymbol.
  - apply Hblock.
    exact
      ((fix go (fields : list value) : Forall P fields :=
          match fields with
          | [] => Forall_nil _
          | field :: fields => Forall_cons _ (IH field) (go fields)
          end)
         fields).
  - apply Harray.
    exact
      ((fix go (elements : list value) : Forall P elements :=
          match elements with
          | [] => Forall_nil _
          | element :: elements => Forall_cons _ (IH element) (go elements)
          end)
         elements).
  - apply Hboxed.
  - apply Hlocation.
  - apply Hpartial; [ apply IH | ].
    exact
      ((fix go (args : list value) : Forall P args :=
          match args with
          | [] => Forall_nil _
          | arg :: args => Forall_cons _ (IH arg) (go args)
          end)
         args).
Qed.

(* What a value is at its head, for the cases of the primitives. *)
Definition location_of (v : value) : option location :=
  match v with
  | Value_location l => Some l
  | _ => None
  end.

Lemma eval_phys_equal_location_cases (st : static_env) (v1 v2 : value) :
  eval_phys_equal_location st v1 v2
  = match location_of v1, location_of v2 with
    | Some l1, Some l2 => Some (Pos.eqb l1 l2)
    | Some _, None =>
      if is_some (int_of_value Standard_tagged_immediate v2)
         || is_allocated st v2 || is_imported_pointer st v2
      then Some false
      else None
    | None, Some _ =>
      if is_some (int_of_value Standard_tagged_immediate v1)
         || is_allocated st v1 || is_imported_pointer st v1
      then Some false
      else None
    | None, None => None
    end.
Proof. destruct v1, v2; reflexivity. Qed.

Lemma exec_bound_prim_unary (st : static_env) (hp : heap)
    (p : unary_primitive) (v : value) :
  exec_bound_prim st hp (Unary p) [v]
  = match location_of v with
    | Some l =>
      let* c := lookup l hp in
      let* result := exec_unary_on_cell p l c in
      Some (result, hp)
    | None =>
      let* result := eval_unary_primitive st p v in
      Some (result, hp)
    end.
Proof. destruct v; reflexivity. Qed.

Lemma exec_block_set (st : static_env) (hp : heap) (kind : block_access_kind)
    (init : init_or_assign) (field : Z) (v new : value) :
  exec_heap_primitive st hp (Block_set kind init field) [v; new]
  = match location_of v with
    | Some l =>
      match lookup l hp with
      | Some (Cell_block Cell_mutable tag fields) =>
        let* fields' := store_field kind field new tag fields in
        Some (unit_value, (l, Cell_block Cell_mutable tag fields') :: hp)
      | _ => None
      end
    | None => None
    end.
Proof. destruct v; reflexivity. Qed.

Lemma exec_array_set (st : static_env) (hp : heap) (akind : array_kind)
    (set_kind : array_set_kind) (v index new : value) :
  exec_heap_primitive st hp (Array_set akind set_kind) [v; index; new]
  = match location_of v with
    | Some l =>
      match lookup l hp with
      | Some (Cell_array kind elements) =>
        let* elements' := array_stored set_kind kind elements index new in
        Some (unit_value, (l, Cell_array kind elements') :: hp)
      | _ => None
      end
    | None => None
    end.
Proof. destruct v; reflexivity. Qed.

Lemma load_field_in (kind : block_access_kind) (field : Z) (tag : Z)
    (fields : list value) (loaded : value) :
  load_field kind field tag fields = Some loaded -> loaded ∈ fields.
Proof.
  destruct kind as [known_tag size field_kind]. simpl.
  destruct (_ && _); [ | discriminate ].
  destruct (nth_error fields (Z.to_nat field)) as [v | ] eqn:Hnth; simpl;
    [ | discriminate ].
  destruct
    match field_kind with
    | Any_value => kind_eqb (kind_of_value v) Value
    | Immediate => is_tagged_immediate v
    end;
    [ | discriminate ].
  intros [= <-]. exact (nth_error_In _ _ Hnth).
Qed.

Lemma value_closed_evolves (h h1 : heap) (v : value) :
  heap_evolves h h1 -> value_closed h v -> value_closed h1 v.
Proof.
  intros Hevolves Hclosed l Hl. specialize (Hclosed l Hl).
  exact (heap_evolves_bound _ _ _ Hevolves Hclosed).
Qed.

Lemma const_closed (h : heap) (c : const) : value_closed h (Value_const c).
Proof. intros l [ ]. Qed.

(* What a closed value holds is closed. *)
Lemma block_fields_closed (st : static_env) (h : heap) (v : value) (tag : Z)
    (fields : list value) (field : value) :
  memory_closed st h ->
  value_closed h v ->
  block_of st v = Some (tag, fields) ->
  field ∈ fields ->
  value_closed h field.
Proof.
  intros [_ Hstatics] Hclosed Hblock Hfield.
  destruct v; simpl in Hblock; try discriminate.
  - destruct (lookup s (static_values st))
      as [[ | | tag0 fields0 | | | | ] | ]
      eqn:Hs; try discriminate.
    injection Hblock as <- <-. exact (Hstatics s _ field Hs Hfield).
  - injection Hblock as <- <-. intros l Hl. apply Hclosed. simpl.
    apply in_flat_map. exists field. auto.
Qed.

Lemma closure_slots_closed (st : static_env) (h : heap) (v : value)
    (code : code_id) (slots : list (value_slot * value)) (w : value) :
  memory_closed st h ->
  value_closed h v ->
  closure_of st v = Some (code, slots) ->
  w ∈ map snd slots ->
  value_closed h w.
Proof.
  intros [_ Hstatics] Hclosed Hclosure Hw.
  destruct v as [c | code0 slots0 | fslot decls slots0 | s | tag fields
    | elements | c | l | callee0 applied applied_arity];
    simpl in Hclosure; try discriminate.
  - injection Hclosure as <- <-. intros l Hl. apply Hclosed. simpl.
    apply in_flat_map. apply in_map_iff in Hw as (slot & <- & Hslot).
    exists slot. auto.
  - destruct (lookup fslot decls); simpl in Hclosure; [ | discriminate ].
    injection Hclosure as <- <-. intros l Hl. apply Hclosed. simpl.
    apply in_flat_map. apply in_map_iff in Hw as (slot & <- & Hslot).
    exists slot. auto.
  - destruct (lookup s (static_values st))
      as [[code0 slots0 | fslot syms decls slots0 | | | | | ] | ]
      eqn:Hs; try discriminate.
    + injection Hclosure as <- <-. exact (Hstatics s _ w Hs Hw).
    + destruct (lookup fslot decls); simpl in Hclosure; [ | discriminate ].
      injection Hclosure as <- <-. exact (Hstatics s _ w Hs Hw).
Qed.

(* A primitive that does not touch the heap makes a closed value of
   closed arguments. *)
Lemma eval_prim_closed (st : static_env) (h : heap) (p : prim)
    (vs : list value) (result : value) :
  memory_closed st h ->
  (∀ v : value, v ∈ vs -> value_closed h v) ->
  eval_prim st p vs = Some result ->
  value_closed h result.
Proof.
  intros Hmemory Hvs Heval.
  apply
    (FlambdaPrimitiveProofs.eval_prim_preserves (value_closed h) st p vs
       result);
    try assumption.
  - intros c. split; intros l [ ].
  - intros tag fields. split.
    + intros Hblock v Hv l Hl. apply Hblock. simpl. apply in_flat_map.
      exists v. auto.
    + intros Hfields l Hl. simpl in Hl.
      apply in_flat_map in Hl as (field & Hfield & Hl).
      exact (Hfields field Hfield l Hl).
  - intros elements. split.
    + intros Harray v Hv l Hl. apply Harray. simpl. apply in_flat_map.
      exists v. auto.
    + intros Helements l Hl. simpl in Hl.
      apply in_flat_map in Hl as (element & Helement & Hl).
      exact (Helements element Helement l Hl).
  - intros s tag fields _ Hs v Hv. exact (proj2 Hmemory s _ v Hs Hv).
  - intros s elements _ Hs v Hv. exact (proj2 Hmemory s _ v Hs Hv).
Qed.

Lemma kind_eqb_true (k k' : flambda_kind) : kind_eqb k k' = true -> k = k'.
Proof. unfold kind_eqb. destruct (kind_eq_dec k k'); congruence. Qed.

Lemma store_field_in (kind : block_access_kind) (field : Z) (new : value)
    (tag : Z) (fields fields1 : list value) :
  store_field kind field new tag fields = Some fields1 ->
  kind_of_value new = Value
  ∧ ∀ y : value, y ∈ fields1 -> y = new ∨ y ∈ fields.
Proof.
  destruct kind as [known_tag size field_kind]. simpl.
  destruct
    (known_is known_tag tag
     && known_is size (Z.of_nat (length fields))
     && (0 <=? field));
    simpl; [ | discriminate ].
  destruct
    match field_kind with
    | Any_value => kind_eqb (kind_of_value new) Value
    | Immediate => is_tagged_immediate new
    end
    eqn:Hcheck; [ | discriminate ].
  intros Hreplace. split.
  - destruct field_kind.
    + exact (kind_eqb_true _ _ Hcheck).
    + destruct new; try discriminate. destruct c; try discriminate.
      reflexivity.
  - intros y Hy.
    exact (FlambdaPrimitiveProofs.replace_nth_in _ _ _ _ _ Hreplace Hy).
Qed.

(* A closed memory stays closed when a cell is bound, at a location that
   is fresh or in place of a cell that evolves to it, if the values of
   the cell are closed and of kind [Value]. *)
Lemma memory_closed_write (st : static_env) (h : heap) (l : location)
    (c : cell) :
  memory_closed st h ->
  heap_evolves h ((l, c) :: h) ->
  (∀ v : value,
    v ∈ values_of_cell c -> value_closed h v ∧ kind_of_value v = Value) ->
  memory_closed st ((l, c) :: h).
Proof.
  intros [Hheap Hstatics] Hevolves Hcell. split.
  - intros l0 c0 v Hl0 Hv. simpl in Hl0.
    destruct (Pos.eqb l0 l).
    + injection Hl0 as <-. destruct (Hcell v Hv) as [Hclosed Hkind].
      split; [ exact (value_closed_evolves _ _ _ Hevolves Hclosed) | ].
      exact Hkind.
    + destruct (Hheap l0 c0 v Hl0 Hv) as [Hclosed Hkind].
      split; [ exact (value_closed_evolves _ _ _ Hevolves Hclosed) | ].
      exact Hkind.
  - intros s a v Hs Hv.
    exact (value_closed_evolves _ _ _ Hevolves (Hstatics s a v Hs Hv)).
Qed.

(* A primitive keeps the memory closed, and its value is closed. *)
Theorem exec_prim_closed : exec_prim_closed_statement.
Proof.
  intros st h h1 p vs result.
  intros Hmemory Hvs Hexec. apply exec_prim_bound in Hexec as [_ Hexec].
  revert Hexec.
  assert (Hpure :
            (let* v := eval_prim st p vs in Some (v, h)) = Some (result, h1) ->
            memory_closed st h1 ∧ value_closed h1 result).
  { destruct (eval_prim st p vs) as [v | ] eqn:Heval; simpl;
      [ | discriminate ].
    intros [= <- <-].
    split;
      [ exact Hmemory | exact (eval_prim_closed _ _ _ _ _ Hmemory Hvs Heval) ].
  }
  destruct Hmemory as [Hheap Hstatics].
  destruct p as [p | p | p | p | rp]; try exact Hpure.
  - destruct vs as [ | v [ | v2 vs]]; try exact Hpure.
    2: { destruct v; simpl; discriminate. }
    rewrite exec_bound_prim_unary.
    destruct (location_of v) as [l | ] eqn:Hlocation.
    2: { destruct v; try discriminate; exact Hpure. }
    destruct v; try discriminate. injection Hlocation as ->.
    destruct (lookup l h) as [c | ] eqn:Hc; simpl; [ | discriminate ].
    destruct (exec_unary_on_cell p l c) as [res | ] eqn:Hres; simpl;
      [ | discriminate ].
    intros [= <- <-]. split; [ split; assumption | ].
    destruct c as [cell_mut tag fields | akind elements].
    + destruct p; simpl in Hres; try discriminate.
      * destruct (load_allowed _ cell_mut); [ | discriminate ].
        apply load_field_in in Hres.
        exact (proj1 (Hheap l _ res Hc Hres)).
      * injection Hres as <-. apply const_closed.
      * injection Hres as <-. apply const_closed.
      * injection Hres as <-. apply const_closed.
      * destruct (kind_eqb _ Value); [ | discriminate ].
        injection Hres as <-. intros l0 [<- | [ ]]. unfold is_bound.
        rewrite Hc. reflexivity.
    + (* An array: [Is_int] without [variant_only], [Is_null], the
         length and the identity. *)
      destruct p as [ | [ | ] | | | length_kind | | okind | | | | | | | | ];
        simpl in Hres; try discriminate.
      * injection Hres as <-. apply const_closed.
      * injection Hres as <-. apply const_closed.
      * unfold array_length_of in Hres.
        destruct (in_range _ _); [ | discriminate ].
        injection Hres as <-. apply const_closed.
      * destruct (kind_eqb _ Value); [ | discriminate ].
        injection Hres as <-. intros l0 [<- | [ ]]. unfold is_bound.
        rewrite Hc. reflexivity.
  - rewrite exec_bound_prim_binary.
    destruct (phys_equal_on_location p vs).
    + destruct p; try discriminate.
      destruct vs as [ | v1 [ | v2 [ | v3 vs]]]; try discriminate.
      destruct (eval_phys_equal_location st v1 v2); simpl;
        [ | discriminate ].
      intros [= <- <-]. split; [ split; assumption | apply const_closed ].
    + destruct (binary_on_cell p vs) as [[l v2] | ]; [ | exact Hpure ].
      destruct (lookup l h) as [c | ] eqn:Hc; simpl; [ | discriminate ].
      destruct (exec_binary_on_cell p l c v2) as [res | ] eqn:Hres; simpl;
        [ | discriminate ].
      intros [= <- <-]. split; [ split; assumption | ].
      (* The load of an array: an element of the cell. *)
      destruct c as [cell_mut tag fields | akind elements];
        [ destruct p; simpl in Hres; discriminate | ].
      destruct p as [array_kind load_kind mut | | | | | ]; simpl in Hres;
        try discriminate.
      destruct (load_allowed mut Cell_mutable); [ | discriminate ].
      apply FlambdaPrimitiveProofs.array_loaded_element in Hres as [Hin _].
      exact (proj1 (Hheap l _ res Hc Hin)).
  - change (exec_bound_prim st h (Heap p) vs)
      with (exec_heap_primitive st h p vs).
    destruct p as [mut tag | kind init field | kind | akind set_kind].
    + simpl. destruct (forallb _ vs) eqn:Hkinds; [ | discriminate ].
      intros [= <- <-].
      set (l := FlambdaHeapSpec.fresh_location h).
      assert (Hevolves :
                heap_evolves h ((l, Cell_block mut tag vs) :: h)).
      { apply heap_evolves_fresh. }
      split; [ split | ].
      * intros l0 c v Hl0 Hv. simpl in Hl0.
        destruct (Pos.eqb l0 l).
        -- injection Hl0 as <-. simpl in Hv. split.
           ++ exact (value_closed_evolves _ _ _ Hevolves (Hvs v Hv)).
           ++ rewrite forallb_forall in Hkinds.
              exact (kind_eqb_true _ _ (Hkinds v Hv)).
        -- destruct (Hheap l0 c v Hl0 Hv) as [Hclosed Hkind].
           split; [ exact (value_closed_evolves _ _ _ Hevolves Hclosed) | ].
           exact Hkind.
      * intros s a v Hs Hv.
        exact (value_closed_evolves _ _ _ Hevolves (Hstatics s a v Hs Hv)).
      * intros l0 [<- | [ ]]. unfold is_bound. simpl.
        rewrite Pos.eqb_refl. reflexivity.
    + destruct vs as [ | v [ | new [ | v3 vs]]]; try (simpl; discriminate).
      rewrite exec_block_set.
      destruct (location_of v) as [l | ]; [ | discriminate ].
      destruct (lookup l h) as [[[ | ] tag fields | akind elements] | ]
        eqn:Hc;
        try discriminate.
      destruct (store_field kind field new tag fields) as [fields1 | ]
        eqn:Hstore; simpl; [ | discriminate ].
      intros [= <- <-].
      destruct (store_field_in _ _ _ _ _ _ Hstore) as [Hkind Hin].
      assert (Hnew : value_closed h new).
      { apply Hvs. right. left. reflexivity. }
      assert (Hevolves :
                heap_evolves h ((l, Cell_block Cell_mutable tag fields1) :: h)).
      { exact
          (heap_evolves_write_block _ _ _ _ _ Hc
             (store_field_length _ _ _ _ _ _ Hstore)). }
      split; [ split | apply const_closed ].
      * intros l0 c v0 Hl0 Hv0. simpl in Hl0.
        destruct (Pos.eqb l0 l).
        -- injection Hl0 as <-. simpl in Hv0.
           destruct (Hin v0 Hv0) as [-> | Hold].
           ++ split;
                [ exact (value_closed_evolves _ _ _ Hevolves Hnew)
                | exact Hkind ].
           ++ destruct (Hheap l _ v0 Hc Hold) as [Hclosed Hkind0].
              split;
                [ exact (value_closed_evolves _ _ _ Hevolves Hclosed)
                | exact Hkind0 ].
        -- destruct (Hheap l0 c v0 Hl0 Hv0) as [Hclosed Hkind0].
           split;
             [ exact (value_closed_evolves _ _ _ Hevolves Hclosed)
             | exact Hkind0 ].
      * intros s a v0 Hs Hv0.
        exact (value_closed_evolves _ _ _ Hevolves (Hstatics s a v0 Hs Hv0)).
    + (* [Make_mutable_array] *)
      simpl. destruct vs as [ | v0 vs0]; [ discriminate | ].
      destruct (array_made kind (v0 :: vs0)) as [elements | ] eqn:Hmade;
        simpl; [ | discriminate ].
      intros [= <- <-].
      apply FlambdaPrimitiveProofs.array_made_of_arguments in Hmade
        as [-> Hok].
      split.
      * apply memory_closed_write;
          [ split; assumption | apply heap_evolves_fresh | ].
        intros v Hv.
        split;
          [ exact (Hvs v Hv)
          | exact (FlambdaPrimitiveProofs.array_element_kind _ _ (Hok v Hv)) ].
      * intros l0 [<- | [ ]]. unfold is_bound. simpl.
        rewrite Pos.eqb_refl. reflexivity.
    + (* [Array_set] *)
      destruct vs as [ | v [ | index [ | new [ | v4 vs]]]];
        try (simpl; discriminate).
      rewrite exec_array_set.
      destruct (location_of v) as [l | ]; [ | discriminate ].
      destruct (lookup l h) as [[mut tag fields | kind elements] | ] eqn:Hc;
        try discriminate.
      destruct (array_stored set_kind kind elements index new)
        as [stored | ] eqn:Hstore; simpl; [ | discriminate ].
      intros [= <- <-].
      destruct
        (FlambdaPrimitiveProofs.array_stored_elements _ _ _ _ _ _ Hstore)
        as (Hlength & Hok & Hin).
      assert (Hnew : value_closed h new).
      { apply Hvs. right. right. left. reflexivity. }
      split; [ | apply const_closed ].
      apply memory_closed_write;
        [ split; assumption
        | exact (heap_evolves_write_array _ _ _ _ _ Hc Hlength) | ].
      intros v0 Hv0. destruct (Hin v0 Hv0) as [-> | Hold].
      * split;
          [ exact Hnew
          | exact (FlambdaPrimitiveProofs.array_element_kind _ _ Hok) ].
      * exact (Hheap l _ v0 Hc Hold).
Qed.

(* ======================================================================== *)
(* The primitives on a heap and [eval_prim]                                 *)
(* ======================================================================== *)

Theorem exec_prim_heap : exec_prim_heap_statement.
Proof.
  intros statics h h' p vs v Hnot Hexec.
  apply exec_prim_bound in Hexec as [_ Hexec].
  exact (exec_bound_prim_not_heap _ _ _ _ _ _ Hnot Hexec).
Qed.

(* A primitive that does not read the heap has no value on a location,
   by [eval_prim] or on a cell. *)
Local Ltac by_cases_to_none :=
  repeat match goal with
         | x : ?T |- _ =>
           lazymatch T with
           | static_env => fail
           | location => fail
           | value => fail
           | Z => fail
           | _ => destruct x
           end
         end;
  reflexivity.

Lemma eval_unary_on_location (statics : static_env) (p : unary_primitive)
    (l : location) :
  prim_reads_heap (Unary p) = false ->
  eval_unary_primitive statics p (Value_location l) = None.
Proof.
  intros Hreads.
  destruct p; try discriminate Hreads; clear Hreads; by_cases_to_none.
Qed.

Lemma exec_unary_on_cell_reads_heap (p : unary_primitive) (l : location)
    (c : cell) :
  prim_reads_heap (Unary p) = false -> exec_unary_on_cell p l c = None.
Proof.
  intros Hreads. destruct c; destruct p; try discriminate Hreads; reflexivity.
Qed.

Lemma eval_binary_on_location (statics : static_env) (p : binary_primitive)
    (l : location) (v2 : value) :
  prim_reads_heap (Binary p) = false ->
  eval_binary_primitive statics p (Value_location l) v2 = None.
Proof.
  intros Hreads.
  destruct p; try discriminate Hreads; clear Hreads; by_cases_to_none.
Qed.

Lemma exec_binary_on_cell_reads_heap (p : binary_primitive) (l : location)
    (c : cell) (v2 : value) :
  prim_reads_heap (Binary p) = false -> exec_binary_on_cell p l c v2 = None.
Proof.
  intros Hreads. destruct c; destruct p; try discriminate Hreads; reflexivity.
Qed.

(* A primitive that does not read the heap is [eval_prim] on arguments
   that are bound. *)
Lemma exec_bound_prim_eval_prim (statics : static_env) (h : heap) (p : prim)
    (vs : list value) :
  prim_reads_heap p = false ->
  exec_bound_prim statics h p vs
  = let* v := eval_prim statics p vs in
    Some (v, h).
Proof.
  intros Hreads.
  destruct p as [p | p | p | q | rp];
    [ | | reflexivity | discriminate Hreads | reflexivity ].
  - destruct vs as [ | v0 [ | v1 vs]];
      [ reflexivity | | destruct v0; reflexivity ].
    rewrite exec_bound_prim_unary.
    destruct (location_of v0) as [l | ] eqn:Hl; [ | reflexivity ].
    destruct v0; try discriminate Hl. injection Hl as ->.
    change (eval_prim statics (Unary p) [Value_location l])
      with (eval_unary_primitive statics p (Value_location l)).
    rewrite (eval_unary_on_location _ _ _ Hreads).
    destruct (lookup l h) as [c | ]; simpl; [ | reflexivity ].
    rewrite (exec_unary_on_cell_reads_heap _ _ _ Hreads). reflexivity.
  - rewrite exec_bound_prim_binary.
    destruct (phys_equal_on_location p vs) eqn:Hphys.
    + destruct p; try discriminate Hphys. discriminate Hreads.
    + destruct (binary_on_cell p vs) as [[l v2] | ] eqn:Hcell;
        [ | reflexivity ].
      assert (Hvs : vs = [Value_location l; v2]).
      { destruct p; try discriminate Hcell;
          destruct vs as [ | v1 [ | v2' [ | v3 vs]]]; try discriminate Hcell;
          destruct v1; try discriminate Hcell;
          injection Hcell as -> ->; reflexivity. }
      subst vs.
      change (eval_prim statics (Binary p) [Value_location l; v2])
        with (eval_binary_primitive statics p (Value_location l) v2).
      rewrite (eval_binary_on_location _ _ _ _ Hreads).
      destruct (lookup l h) as [c | ]; simpl; [ | reflexivity ].
      rewrite (exec_binary_on_cell_reads_heap _ _ _ _ Hreads). reflexivity.
Qed.

Theorem exec_prim_eval_prim : exec_prim_eval_prim_statement.
Proof.
  intros statics h h' p vs v Hreads Hexec.
  apply exec_prim_bound in Hexec as [_ Hexec].
  rewrite (exec_bound_prim_eval_prim _ _ _ _ Hreads) in Hexec.
  destruct (eval_prim statics p vs) as [v0 | ]; [ | discriminate ].
  injection Hexec as <- <-. split; reflexivity.
Qed.

Theorem eval_prim_exec_prim : eval_prim_exec_prim_statement.
Proof.
  intros statics h p vs v Hreads Hbound Heval.
  unfold exec_prim.
  assert (Hall : forallb (argument_bound statics h) vs = true).
  { apply forallb_forall. intros arg Harg.
    destruct arg as [c | code slots | fslot decls slots | s | tag fields
                    | elements | c | l | callee applied applied_arity];
      try reflexivity.
    - simpl. unfold is_bound.
      pose proof
        (FlambdaPrimitiveProofs.eval_prim_symbols_bound _ _ _ _ _ Heval Harg)
        as Hs.
      destruct (lookup s (static_values statics)); [ reflexivity | ].
      contradiction Hs. reflexivity.
    - exact (Hbound l Harg). }
  rewrite Hall, (exec_bound_prim_eval_prim _ _ _ _ Hreads), Heval.
  reflexivity.
Qed.

(* A primitive that is not of the heap is [eval_prim] on arguments that
   are bound and none of which is a location. *)
Lemma exec_bound_prim_no_location (statics : static_env) (h : heap)
    (p : prim) (vs : list value) :
  (∀ q : heap_primitive, p <> Heap q) ->
  (∀ l : location, Value_location l ∉ vs) ->
  exec_bound_prim statics h p vs
  = let* v := eval_prim statics p vs in
    Some (v, h).
Proof.
  intros Hnot Hnone.
  assert (Hfirst :
            ∀ (l : location) (rest : list value),
              vs = Value_location l :: rest -> False).
  { intros l rest ->. apply (Hnone l). left. reflexivity. }
  assert (Hsecond :
            ∀ (v1 : value) (l : location) (rest : list value),
              vs = v1 :: Value_location l :: rest -> False).
  { intros v1 l rest ->. apply (Hnone l). right. left. reflexivity. }
  destruct p as [p | p | p | q | rp];
    [ | | reflexivity | contradiction (Hnot q eq_refl) | reflexivity ].
  - destruct vs as [ | v0 [ | v1 vs]];
      [ reflexivity | | destruct v0; reflexivity ].
    destruct v0; try reflexivity.
    contradiction (Hfirst _ _ eq_refl).
  - destruct vs as [ | v1 [ | v2 [ | v3 vs]]].
    + destruct p; reflexivity.
    + destruct v1; destruct p; reflexivity.
    + destruct v1; try contradiction (Hfirst _ _ eq_refl);
        destruct v2; try contradiction (Hsecond _ _ _ eq_refl);
        destruct p; reflexivity.
    + destruct v1; destruct p; reflexivity.
Qed.

Theorem exec_prim_no_location : exec_prim_no_location_statement.
Proof.
  intros statics h p vs Hnot Hnone.
  unfold exec_prim.
  destruct (forallb (argument_bound statics h) vs) eqn:Hall.
  - exact (exec_bound_prim_no_location _ _ _ _ Hnot Hnone).
  - (* An argument that is not bound is a symbol, and [eval_prim] has no
       value on it. *)
    destruct (eval_prim statics p vs) as [v | ] eqn:Heval; [ | reflexivity ].
    exfalso.
    assert (Hbound : forallb (argument_bound statics h) vs = true).
    { apply forallb_forall. intros arg Harg.
      destruct arg as [c | code slots | fslot decls slots | s | tag fields
                      | elements | c | l | callee applied applied_arity];
        try reflexivity.
      - simpl. unfold is_bound.
        pose proof
          (FlambdaPrimitiveProofs.eval_prim_symbols_bound _ _ _ _ _ Heval
             Harg)
          as Hs.
        destruct (lookup s (static_values statics)); [ reflexivity | ].
        contradiction Hs. reflexivity.
      - contradiction (Hnone l Harg). }
    rewrite Hbound in Hall. discriminate Hall.
Qed.

(* ======================================================================== *)
(* The values that a heap holds                                             *)
(* ======================================================================== *)

Theorem exec_prim_preserves : exec_prim_preserves_statement.
Proof.
  intros P st h h1 p vs result Hnumbers Hblocks Hof_arrays Hstatic Harrays
    Hlocations Hvs Hheap Hexec.
  apply exec_prim_bound in Hexec as [_ Hexec]. revert Hexec.
  assert (Hpure :
            (let* v := eval_prim st p vs in Some (v, h)) = Some (result, h1) ->
            P result ∧ heap_values_all P h1).
  { destruct (eval_prim st p vs) as [v | ] eqn:Heval; simpl;
      [ | discriminate ].
    intros [= <- <-]. split; [ | exact Hheap ].
    exact
      (FlambdaPrimitiveProofs.eval_prim_preserves P st p vs v Hnumbers Hblocks
         Hof_arrays Hstatic Harrays Hvs Heval). }
  assert (Hconst : ∀ c : const, P (Value_const c)).
  { intros c. exact (proj1 (Hnumbers c)). }
  assert (Hcell :
            ∀ (l : location) (c : cell) (v : value),
              lookup l h = Some c -> v ∈ values_of_cell c -> P v).
  { intros l c v Hl Hv. exact (Hheap l c v (lookup_in _ _ _ Hl) Hv). }
  assert (Hwrite :
            ∀ (l : location) (c : cell),
              (∀ v : value, v ∈ values_of_cell c -> P v) ->
              heap_values_all P ((l, c) :: h)).
  { intros l c Hc l0 c0 v [[= <- <-] | Hin] Hv;
      [ exact (Hc v Hv) | exact (Hheap l0 c0 v Hin Hv) ]. }
  destruct p as [p | p | p | p | rp]; try exact Hpure.
  - destruct vs as [ | v [ | v2 vs]]; try exact Hpure.
    2: { destruct v; simpl; discriminate. }
    rewrite exec_bound_prim_unary.
    destruct (location_of v) as [l | ] eqn:Hlocation.
    2: { destruct v; try discriminate; exact Hpure. }
    destruct v; try discriminate. injection Hlocation as ->.
    destruct (lookup l h) as [c | ] eqn:Hc; simpl; [ | discriminate ].
    destruct (exec_unary_on_cell p l c) as [res | ] eqn:Hres; simpl;
      [ | discriminate ].
    intros [= <- <-]. split; [ | exact Hheap ].
    destruct c as [cell_mut tag fields | akind elements].
    + destruct p; simpl in Hres; try discriminate.
      * destruct (load_allowed _ cell_mut); [ | discriminate ].
        apply load_field_in in Hres. exact (Hcell l _ res Hc Hres).
      * injection Hres as <-. apply Hconst.
      * injection Hres as <-. apply Hconst.
      * injection Hres as <-. apply Hconst.
      * destruct (kind_eqb _ Value); [ | discriminate ].
        injection Hres as <-. apply Hlocations.
    + destruct p as [ | [ | ] | | | length_kind | | okind | | | | | | | | ];
        simpl in Hres; try discriminate.
      * injection Hres as <-. apply Hconst.
      * injection Hres as <-. apply Hconst.
      * unfold array_length_of in Hres.
        destruct (in_range _ _); [ | discriminate ].
        injection Hres as <-. apply Hconst.
      * destruct (kind_eqb _ Value); [ | discriminate ].
        injection Hres as <-. apply Hlocations.
  - rewrite exec_bound_prim_binary.
    destruct (phys_equal_on_location p vs).
    + destruct p; try discriminate.
      destruct vs as [ | v1 [ | v2 [ | v3 vs]]]; try discriminate.
      destruct (eval_phys_equal_location st v1 v2); simpl;
        [ | discriminate ].
      intros [= <- <-]. split; [ apply Hconst | exact Hheap ].
    + destruct (binary_on_cell p vs) as [[l v2] | ]; [ | exact Hpure ].
      destruct (lookup l h) as [c | ] eqn:Hc; simpl; [ | discriminate ].
      destruct (exec_binary_on_cell p l c v2) as [res | ] eqn:Hres; simpl;
        [ | discriminate ].
      intros [= <- <-]. split; [ | exact Hheap ].
      destruct c as [cell_mut tag fields | akind elements];
        [ destruct p; simpl in Hres; discriminate | ].
      destruct p as [array_kind load_kind mut | | | | | ]; simpl in Hres;
        try discriminate.
      destruct (load_allowed mut Cell_mutable); [ | discriminate ].
      apply FlambdaPrimitiveProofs.array_loaded_element in Hres as [Hin _].
      exact (Hcell l _ res Hc Hin).
  - change (exec_bound_prim st h (Heap p) vs)
      with (exec_heap_primitive st h p vs).
    destruct p as [mut tag | kind init field | kind | akind set_kind].
    + simpl. destruct (forallb _ vs) eqn:Hkinds; [ | discriminate ].
      intros [= <- <-]. split; [ apply Hlocations | ].
      apply Hwrite. exact Hvs.
    + destruct vs as [ | v [ | new [ | v3 vs]]]; try (simpl; discriminate).
      rewrite exec_block_set.
      destruct (location_of v) as [l | ]; [ | discriminate ].
      destruct (lookup l h) as [[[ | ] tag fields | akind elements] | ]
        eqn:Hc;
        try discriminate.
      destruct (store_field kind field new tag fields) as [fields1 | ]
        eqn:Hstore; simpl; [ | discriminate ].
      intros [= <- <-].
      destruct (store_field_in _ _ _ _ _ _ Hstore) as [_ Hin].
      split; [ apply Hconst | ].
      apply Hwrite. intros v0 Hv0. simpl in Hv0.
      destruct (Hin v0 Hv0) as [-> | Hold].
      * apply Hvs. right. left. reflexivity.
      * exact (Hcell l _ v0 Hc Hold).
    + simpl. destruct vs as [ | v0 vs0]; [ discriminate | ].
      destruct (array_made kind (v0 :: vs0)) as [elements | ] eqn:Hmade;
        simpl; [ | discriminate ].
      intros [= <- <-].
      apply FlambdaPrimitiveProofs.array_made_of_arguments in Hmade
        as [-> Hok].
      split; [ apply Hlocations | ].
      apply Hwrite. exact Hvs.
    + destruct vs as [ | v [ | index [ | new [ | v4 vs]]]];
        try (simpl; discriminate).
      rewrite exec_array_set.
      destruct (location_of v) as [l | ]; [ | discriminate ].
      destruct (lookup l h) as [[mut tag fields | kind elements] | ] eqn:Hc;
        try discriminate.
      destruct (array_stored set_kind kind elements index new)
        as [stored | ] eqn:Hstore; simpl; [ | discriminate ].
      intros [= <- <-].
      destruct
        (FlambdaPrimitiveProofs.array_stored_elements _ _ _ _ _ _ Hstore)
        as (_ & _ & Hin).
      split; [ apply Hconst | ].
      apply Hwrite. intros v0 Hv0. simpl in Hv0.
      destruct (Hin v0 Hv0) as [-> | Hold].
      * apply Hvs. right. right. left. reflexivity.
      * exact (Hcell l _ v0 Hc Hold).
Qed.

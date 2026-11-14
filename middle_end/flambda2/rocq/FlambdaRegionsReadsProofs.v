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

(* Lemmas for the proofs of the statements of
   FlambdaRegionsInvariantSpec.v: the values that a value, and what a
   symbol is bound to, hold directly; what a named expression that
   does not allocate reads, by [eval_named] and on a heap; what an
   allocation and a static constant make; what a primitive of the heap
   leaves in the heap. Nothing here needs review. *)

From Stdlib Require Import ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaPrimitiveModel FlambdaPrimitiveSpec
  FlambdaPrimitiveProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaClosuresProofs FlambdaRegionsSpec FlambdaRegionsProofs
  FlambdaRegionsInvariantSpec.
From OxCamlFlambda2 Require FlambdaHeapProofs.
Import ListNotations.

(* ======================================================================== *)
(* Lists                                                                    *)
(* ======================================================================== *)

Lemma forallb_pairs {A B : Type} (f : B -> bool) (l : list (A * B)) :
  forallb (fun '(_, b) => f b) l = forallb f (map snd l).
Proof.
  induction l as [ | [a b] l IH]; simpl; [ reflexivity | ].
  rewrite IH. reflexivity.
Qed.

Lemma forallb_flat_map {A B : Type} (f : B -> bool) (g : A -> list B)
    (l : list A) :
  forallb f (flat_map g l) = forallb (fun a => forallb f (g a)) l.
Proof.
  induction l as [ | a l IH]; simpl; [ reflexivity | ].
  rewrite forallb_app, IH. reflexivity.
Qed.

Lemma forallb_in {A : Type} (f : A -> bool) (l : list A) (a : A) :
  forallb f l = true -> In a l -> f a = true.
Proof. intros Hall. exact (proj1 (forallb_forall f l) Hall a). Qed.

Lemma forallb_impl {A : Type} (f g : A -> bool) (l : list A) :
  (∀ a : A, In a l -> f a = true -> g a = true) ->
  forallb f l = true -> forallb g l = true.
Proof.
  intros Himpl Hall. apply forallb_forall. intros a Hin.
  exact (Himpl a Hin (forallb_in f l a Hall Hin)).
Qed.

Lemma in_is_some_lookup {A : Type} (name : positive) (a : A)
    (bindings : list (positive * A)) :
  In (name, a) bindings -> is_some (lookup name bindings) = true.
Proof.
  induction bindings as [ | [name' a'] bindings IH]; [ contradiction | ].
  rewrite lookup_cons. intros [[= -> ->] | Hin].
  - rewrite Pos.eqb_refl. reflexivity.
  - destruct (Pos.eqb name name'); [ reflexivity | exact (IH Hin) ].
Qed.

Lemma is_some_lookup_app {A : Type} (name : positive)
    (bindings bindings' : list (positive * A)) :
  is_some (lookup name (bindings ++ bindings'))
  = is_some (lookup name bindings) || is_some (lookup name bindings').
Proof.
  rewrite lookup_app. destruct (lookup name bindings); reflexivity.
Qed.

(* The values of a list of simples are each the value of one of them. *)
Lemma eval_simples_in (vars : env) (args : list simple) :
  ∀ (vs : list value) (w : value),
    eval_simples vars args = Some vs ->
    In w vs ->
    ∃ s : simple, In s args ∧ eval_simple vars s = Some w.
Proof.
  induction args as [ | s args IH]; intros vs w; simpl.
  - intros [= <-] [].
  - destruct (eval_simple vars s) as [v | ] eqn:Hs; simpl; [ | discriminate ].
    destruct (eval_simples vars args) as [vs' | ]; simpl; [ | discriminate ].
    intros [= <-] [<- | Hin].
    + exists s. split; [ left; reflexivity | exact Hs ].
    + destruct (IH vs' w eq_refl Hin) as (s' & Hs' & Heval).
      exists s'. split; [ right; exact Hs' | exact Heval ].
Qed.

Lemma eval_value_slots_in (vars : env)
    (value_slots : list (value_slot * simple)) :
  ∀ (vs : list (value_slot * value)) (w : value),
    eval_value_slots vars value_slots = Some vs ->
    In w (map snd vs) ->
    ∃ s : simple, In s (map snd value_slots) ∧ eval_simple vars s = Some w.
Proof.
  induction value_slots as [ | [slot s] value_slots IH]; intros vs w; simpl.
  - intros [= <-] [].
  - destruct (eval_simple vars s) as [v | ] eqn:Hs; simpl; [ | discriminate ].
    destruct (eval_value_slots vars value_slots) as [vs' | ]; simpl;
      [ | discriminate ].
    intros [= <-] [<- | Hin].
    + exists s. split; [ left; reflexivity | exact Hs ].
    + destruct (IH vs' w eq_refl Hin) as (s' & Hs' & Heval).
      exists s'. split; [ right; exact Hs' | exact Heval ].
Qed.

Lemma eval_simples_forall (P : value -> Prop) (vars : env)
    (args : list simple) (vs : list value) :
  (∀ (s : simple) (w : value),
    In s args -> eval_simple vars s = Some w -> P w) ->
  eval_simples vars args = Some vs ->
  Forall P vs.
Proof.
  intros Hargs Heval. apply Forall_forall. intros w Hin.
  destruct (eval_simples_in vars args vs w Heval Hin) as (s & Hs & Hw).
  exact (Hargs s w Hs Hw).
Qed.

(* ======================================================================== *)
(* The values that a value, and what a symbol is bound to, holds directly   *)
(* ======================================================================== *)

Definition value_parts (v : value) : list value :=
  match v with
  | Value_closure _ value_slots | Value_closure_of_set _ _ value_slots =>
    map snd value_slots
  | Value_block _ fields => fields
  | Value_immutable_array elements => elements
  | Value_boxed_number c => [Value_const c]
  | _ => []
  end.

(* The values that an allocation makes. *)
Definition is_made (v : value) : bool :=
  match v with
  | Value_closure _ _ | Value_closure_of_set _ _ _ | Value_block _ _
  | Value_immutable_array _ | Value_boxed_number _ =>
    true
  | _ => false
  end.

Definition allocated_values (allocated : statically_allocated_value)
    : list value :=
  match allocated with
  | Static_closure _ value_slots
  | Static_closure_of_set _ _ _ value_slots =>
    map snd value_slots
  | Static_block _ fields => fields
  | Static_boxed_number contents => [contents]
  | Static_string _ | Static_imported _ => []
  | Static_immutable_array elements => elements
  end.

Definition allocated_symbols (allocated : statically_allocated_value)
    : list symbol :=
  match allocated with
  | Static_closure_of_set _ closure_symbols _ _ => map snd closure_symbols
  | _ => []
  end.

Lemma allocated_holds_parts (ok : symbol -> bool)
    (allocated : statically_allocated_value) :
  allocated_holds ok allocated
  = forallb (value_holds ok) (allocated_values allocated)
    && forallb ok (allocated_symbols allocated).
Proof.
  destruct allocated; simpl; rewrite ?forallb_pairs, ?andb_true_r;
    try reflexivity.
  apply andb_comm.
Qed.

Lemma allocated_placed_parts (near ok : symbol -> bool)
    (allocated : statically_allocated_value) :
  allocated_placed near ok allocated
  = forallb (value_placed near ok) (allocated_values allocated)
    && forallb near (allocated_symbols allocated).
Proof.
  destruct allocated; simpl; rewrite ?forallb_pairs, ?andb_true_r;
    try reflexivity.
  apply andb_comm.
Qed.

Lemma allocated_below_parts (next : symbol)
    (allocated : statically_allocated_value) :
  allocated_below next allocated
  = forallb (value_below next) (allocated_values allocated)
    && forallb (symbol_below next) (allocated_symbols allocated).
Proof.
  destruct allocated; simpl; rewrite ?forallb_pairs, ?andb_true_r;
    try reflexivity.
  apply andb_comm.
Qed.

Lemma allocated_no_region_poison_parts
    (allocated : statically_allocated_value) :
  allocated_no_region_poison allocated
  = forallb value_no_region_poison (allocated_values allocated).
Proof.
  destruct allocated; simpl; rewrite ?forallb_pairs, ?andb_true_r;
    reflexivity.
Qed.

Lemma value_holds_parts (ok : symbol -> bool) (v : value) :
  value_holds ok v = true ->
  forallb (value_holds ok) (value_parts v) = true.
Proof.
  destruct v; simpl; rewrite ?forallb_pairs; auto.
Qed.

Lemma value_holds_made (ok : symbol -> bool) (v : value) :
  is_made v = true ->
  forallb (value_holds ok) (value_parts v) = true ->
  value_holds ok v = true.
Proof.
  destruct v; simpl; rewrite ?forallb_pairs; try discriminate; auto.
Qed.

Lemma value_below_parts (next : symbol) (v : value) :
  value_below next v = true ->
  forallb (value_below next) (value_parts v) = true.
Proof.
  destruct v; simpl; rewrite ?forallb_pairs; auto.
Qed.

Lemma value_below_made (next : symbol) (v : value) :
  is_made v = true ->
  forallb (value_below next) (value_parts v) = true ->
  value_below next v = true.
Proof.
  destruct v; simpl; rewrite ?forallb_pairs; try discriminate; auto.
Qed.

Lemma value_no_region_poison_parts (v : value) :
  value_no_region_poison v = true ->
  forallb value_no_region_poison (value_parts v) = true.
Proof.
  destruct v; simpl; rewrite ?forallb_pairs, ?andb_true_r; auto.
Qed.

Lemma value_no_region_poison_made (v : value) :
  is_made v = true ->
  forallb value_no_region_poison (value_parts v) = true ->
  value_no_region_poison v = true.
Proof.
  destruct v; simpl; rewrite ?forallb_pairs, ?andb_true_r;
    try discriminate; auto.
Qed.

(* ======================================================================== *)
(* What a named expression that does not allocate reads                     *)
(* ======================================================================== *)

Lemma const_of_int_no_region_poison (kind : standard_int) (i : Z) :
  const_no_region_poison (const_of_int kind i) = true.
Proof. destruct kind; reflexivity. Qed.

Lemma is_number_of_kind_no_region_poison (kind : boxable_number)
    (c : const) :
  is_number_of_kind kind c = true -> const_no_region_poison c = true.
Proof. destruct c; try reflexivity. destruct kind; discriminate. Qed.

(* The field that a [Block_load] reads from a cell is one of its
   fields. *)
Lemma load_field_in (kind : block_access_kind) (field : Z) (tag : Z)
    (fields : list value) (v : value) :
  load_field kind field tag fields = Some v -> In v fields.
Proof.
  unfold load_field. destruct kind as [ktag size field_kind].
  destruct
    (known_is ktag tag && known_is size (Z.of_nat (length fields))
     && (0 <=? field)%Z);
    [ | discriminate ].
  destruct (nth_error fields (Z.to_nat field)) as [loaded | ] eqn:Hnth;
    [ | discriminate ].
  simpl.
  destruct
    (match field_kind with
     | Any_value => kind_eqb (kind_of_value loaded) Value
     | Immediate => is_tagged_immediate loaded
     end);
    [ | discriminate ].
  intros [= <-]. exact (nth_error_In _ _ Hnth).
Qed.

(* Where the value of a primitive that is not of the heap comes from,
   on a heap: it is the value that [eval_prim] gives; or a value of the
   cell of the location that is the first argument of a load, of a
   block or of an array; or the location that is the argument of an
   [Opaque_identity]; or a constant that is not the poison of the kind
   [Region]. *)
Lemma exec_prim_reads_cases (statics : static_env) (hp hp' : heap) (p : prim)
    (vs : list value) (v : value) :
  (∀ q : heap_primitive, p <> Heap q) ->
  exec_prim statics hp p vs = Some (v, hp') ->
  eval_prim statics p vs = Some v
  ∨ (∃ (l : location) (c : cell),
      lookup l hp = Some c
      ∧ In v (values_of_cell c)
      ∧ ((∃ (access : block_access_kind) (mut : mutability) (field : Z),
           p = Unary (Block_load access mut field)
           ∧ vs = [Value_location l])
         ∨ (∃ (kind : array_kind) (load_kind : array_load_kind)
              (mut : mutability) (v2 : value),
             p = Binary (Array_load kind load_kind mut)
             ∧ vs = [Value_location l; v2])))
  ∨ (∃ (l : location) (kind : flambda_kind),
      v = Value_location l
      ∧ p = Unary (Opaque_identity kind)
      ∧ vs = [Value_location l])
  ∨ (∃ c : const, v = Value_const c ∧ const_no_region_poison c = true).
Proof.
  intros Hnot. unfold exec_prim.
  destruct (forallb (argument_bound statics hp) vs); [ | discriminate ].
  assert (Hpure :
    (let* v0 := eval_prim statics p vs in Some (v0, hp)) = Some (v, hp') ->
    eval_prim statics p vs = Some v).
  { destruct (eval_prim statics p vs) as [v0 | ]; [ | discriminate ].
    simpl. intros [= <- _]. reflexivity. }
  unfold exec_bound_prim.
  destruct p as [u | b | variadic | q | rp].
  - destruct vs as [ | v1 [ | v2 rest]];
      try (intros H; left; exact (Hpure H));
      try (destruct v1; intros H; left; exact (Hpure H)).
    destruct v1 as [ | | | | | | | l | ];
      try (intros H; left; exact (Hpure H)).
    destruct (lookup l hp) as [c | ] eqn:Hl; [ | discriminate ]. simpl.
    destruct (exec_unary_on_cell u l c) as [v0 | ] eqn:Hu;
      [ | discriminate ].
    simpl. intros [= <- _]. unfold exec_unary_on_cell in Hu.
    destruct c as [mut tag fields | kind elements].
    + destruct u; try discriminate Hu.
      * destruct (load_allowed mut0 mut); [ | discriminate Hu ].
        right. left. exists l, (Cell_block mut tag fields).
        split; [ exact Hl | ].
        split; [ exact (load_field_in _ _ _ _ _ Hu) | ].
        left. eexists _, _, _. split; reflexivity.
      * injection Hu as <-. right. right. right.
        eexists. split; reflexivity.
      * injection Hu as <-. right. right. right.
        eexists. split; reflexivity.
      * injection Hu as <-. right. right. right.
        eexists. split; reflexivity.
      * destruct (kind_eqb kind Value); [ | discriminate Hu ].
        injection Hu as <-. right. right. left. exists l, kind.
        split; [ reflexivity | split; reflexivity ].
    + destruct u; try discriminate Hu.
      * destruct variant_only; [ discriminate Hu | ].
        injection Hu as <-. right. right. right.
        eexists. split; reflexivity.
      * injection Hu as <-. right. right. right.
        eexists. split; reflexivity.
      * unfold array_length_of in Hu.
        destruct
          (in_range (int_width Standard_tagged_immediate)
             (Z.of_nat (length elements)));
          [ | discriminate Hu ].
        injection Hu as <-. right. right. right.
        eexists. split; reflexivity.
      * destruct (kind_eqb kind0 Value); [ | discriminate Hu ].
        injection Hu as <-. right. right. left. exists l, kind0.
        split; [ reflexivity | split; reflexivity ].
  - assert (Hcell :
      ∀ (b : binary_primitive) (v2 : value) (l : location),
        (let* c := lookup l hp in
         let* v0 := exec_binary_on_cell b l c v2 in
         Some (v0, hp))
        = Some (v, hp') ->
        ∃ c : cell,
          lookup l hp = Some c
          ∧ In v (values_of_cell c)
          ∧ ∃ (kind : array_kind) (load_kind : array_load_kind)
              (mut : mutability),
              b = Array_load kind load_kind mut).
    { intros b0 v2 l H.
      destruct (lookup l hp) as [c | ] eqn:Hl; [ | discriminate ].
      simpl in H.
      destruct (exec_binary_on_cell b0 l c v2) as [v0 | ] eqn:Hb;
        [ | discriminate ].
      simpl in H. injection H as <- _. exists c.
      split; [ reflexivity | ].
      unfold exec_binary_on_cell in Hb.
      destruct c as [mut tag fields | kind elements];
        [ discriminate Hb | ].
      destruct b0; try discriminate Hb.
      destruct (load_allowed mut Cell_mutable); [ | discriminate Hb ].
      split; [ exact (proj1 (array_loaded_element _ _ _ _ Hb)) | ].
      eexists _, _, _. reflexivity. }
    destruct b as [kind load_kind mut | width | op | kind op | kind op
                  | kind behaviour];
      destruct vs as [ | v1 [ | v2 [ | v3 rest]]];
      try (intros H; left; exact (Hpure H));
      try (destruct v1; intros H; left; exact (Hpure H));
      try (destruct v1 as [ | | | | | | | l | ];
           try (intros H; left; exact (Hpure H));
           intros H;
           destruct (Hcell _ _ l H) as (c0 & Hl0 & Hv0 & k0 & lk0 & m0 & Hb0);
           right; left; exists l, c0;
           split; [ exact Hl0 | ]; split; [ exact Hv0 | ];
           right; exists k0, lk0, m0, v2;
           split; [ f_equal; exact Hb0 | reflexivity ]).
    destruct v1;
      (destruct (is_location _ || is_location v2);
       [ destruct (eval_phys_equal_location statics _ v2) as [same | ];
         [ | discriminate ];
         simpl; intros [= <- _]; right; right; right; eexists;
         split; reflexivity
       | intros H; left; exact (Hpure H) ]).
  - intros H. left. exact (Hpure H).
  - contradiction (Hnot q eq_refl).
  - intros H. left. exact (Hpure H).
Qed.

(* What a primitive of the heap gives: a location that the heap does
   not bind, or the unit value. *)
Lemma exec_prim_heap_value (statics : static_env) (hp hp' : heap)
    (q : heap_primitive) (vs : list value) (v : value) :
  exec_prim statics hp (Heap q) vs = Some (v, hp') ->
  v = Value_location (fresh_location hp) ∨ v = unit_value.
Proof.
  intros Hexec.
  destruct (FlambdaHeapProofs.exec_prim_adds _ _ _ _ _ _ Hexec)
    as [(c & _ & -> & _) | (l & rest & new & c & c' & i & _ & _ & _ & -> & _)].
  - left. reflexivity.
  - right. reflexivity.
Qed.

(* And what it leaves in the heap: a property that holds of the values
   of the heap and of the arguments holds of the values of the heap
   after. *)
Lemma exec_prim_heap_cells (Q : value -> Prop) (statics : static_env)
    (hp hp' : heap) (p : prim) (vs : list value) (v : value) :
  heap_values_all Q hp ->
  (∀ arg : value, In arg vs -> Q arg) ->
  exec_prim statics hp p vs = Some (v, hp') ->
  heap_values_all Q hp'.
Proof.
  intros Hcells Hargs Hexec.
  assert (Hcases :
    (∃ q : heap_primitive, p = Heap q) ∨ (∀ q : heap_primitive, p <> Heap q)).
  { destruct p; first [ left; eexists; reflexivity | right; discriminate ]. }
  destruct Hcases as [(q & ->) | Hnot].
  - destruct (FlambdaHeapProofs.exec_prim_adds _ _ _ _ _ _ Hexec)
      as [(c & Hvalues & _ & ->)
         | (l & rest & new & c & c' & i & -> & Hl & Hwith & _ & ->)].
    + intros l0 c0 w [[= <- <-] | Hin] Hw.
      * rewrite Hvalues in Hw. exact (Hargs w Hw).
      * exact (Hcells l0 c0 w Hin Hw).
    + intros l0 c0 w [[= <- <-] | Hin] Hw.
      * destruct (FlambdaHeapProofs.cell_with_values _ _ _ _ w Hwith Hw)
          as [-> | Hold].
        -- apply Hargs. right. apply in_or_app. right. left. reflexivity.
        -- exact (Hcells l c w (lookup_in _ _ _ Hl) Hold).
      * exact (Hcells l0 c0 w Hin Hw).
  - rewrite (FlambdaHeapProofs.exec_prim_heap _ _ _ _ _ _ Hnot Hexec).
    exact Hcells.
Qed.

(* A property of values that holds of the constants which are not the
   poison of the kind [Region], of the values that a value with the
   property holds directly, and of those that a symbol with the
   property is bound to, holds of the value of a named expression that
   does not allocate, if it holds of the values of its simples. *)
Section Reads.

Variable P : value -> Prop.
Variable statics : static_env.

Hypothesis P_const :
  ∀ c : const, const_no_region_poison c = true -> P (Value_const c).
Hypothesis P_parts : ∀ v w : value, P v -> In w (value_parts v) -> P w.
Hypothesis P_move :
  ∀ (slot slot' : function_slot)
    (function_decls : list (function_slot * code_id))
    (value_slots : list (value_slot * value)),
    P (Value_closure_of_set slot function_decls value_slots) ->
    P (Value_closure_of_set slot' function_decls value_slots).
Hypothesis P_static :
  ∀ (s : symbol) (allocated : statically_allocated_value),
    P (Value_symbol s) ->
    lookup s (static_values statics) = Some allocated ->
    (∀ w : value, In w (allocated_values allocated) -> P w)
    ∧ (∀ s' : symbol,
        In s' (allocated_symbols allocated) -> P (Value_symbol s')).

Lemma block_of_reads (arg : value) (tag : Z) (fields : list value) :
  P arg ->
  block_of statics arg = Some (tag, fields) ->
  ∀ v : value, In v fields -> P v.
Proof.
  intros Harg Hblock v Hin.
  destruct (block_of_cases statics arg tag fields Hblock)
    as [-> | (s & -> & Hlookup)].
  - exact (P_parts _ v Harg Hin).
  - exact (proj1 (P_static s _ Harg Hlookup) v Hin).
Qed.

Lemma array_of_reads (arg : value) (elements : list value) :
  P arg ->
  array_of statics arg = Some elements ->
  ∀ v : value, In v elements -> P v.
Proof.
  intros Harg Harray v Hin.
  destruct (array_of_cases statics arg elements Harray)
    as [-> | (s & -> & Hlookup)].
  - exact (P_parts _ v Harg Hin).
  - exact (proj1 (P_static s _ Harg Hlookup) v Hin).
Qed.

Lemma reads_fields : P_fields P statics.
Proof.
  intros size tuple fields Htuple Hfields.
  apply (tuple_fields_of_blocks P statics size tuple fields);
    [ | exact Hfields ].
  intros tag block Hblock. apply Forall_forall.
  exact (block_of_reads tuple tag block Htuple Hblock).
Qed.

Lemma eval_unary_primitive_reads (p : unary_primitive) (arg v : value) :
  (∀ kind : boxable_number, p <> Box_number kind) ->
  P arg ->
  eval_unary_primitive statics p arg = Some v ->
  P v.
Proof.
  intros Hnot Harg.
  destruct p as [access mut field | variant_only | | | length_kind | | kind
                | kind op | src dst | | kind | kind | | | reinterpret];
    cbn [eval_unary_primitive].
  - unfold eval_block_load. destruct access as [tag size field_kind].
    destruct (block_of statics arg) as [[block_tag fields] | ] eqn:Hblock;
      [ | discriminate ].
    destruct
      (known_is tag block_tag && known_is size (Z.of_nat (length fields))
       && (0 <=? field)%Z);
      [ | discriminate ].
    destruct (nth_error fields (Z.to_nat field)) as [loaded | ] eqn:Hnth;
      [ | discriminate ].
    pose proof
      (block_of_reads arg block_tag fields Harg Hblock loaded
         (nth_error_In _ _ Hnth))
      as Hloaded.
    destruct
      (match field_kind with
       | Any_value => kind_eqb (kind_of_value loaded) Value
       | Immediate => is_tagged_immediate loaded
       end);
      [ | discriminate ].
    intros [= <-]. exact Hloaded.
  - destruct (eval_is_int statics variant_only arg); [ | discriminate ].
    intros [= <-]. apply P_const. reflexivity.
  - destruct (eval_is_null statics arg); [ | discriminate ].
    intros [= <-]. apply P_const. reflexivity.
  - destruct (block_of statics arg) as [[tag fields] | ]; [ | discriminate ].
    intros [= <-]. apply P_const. reflexivity.
  - destruct (array_of statics arg) as [elements | ]; [ | discriminate ].
    unfold array_length_of. destruct (in_range _ _); [ | discriminate ].
    intros [= <-]. apply P_const. reflexivity.
  - destruct (string_of statics arg) as [bytes | ]; [ | discriminate ].
    destruct (in_range _ _); [ | discriminate ].
    intros [= <-]. apply P_const. reflexivity.
  - destruct (kind_eqb (kind_of_value arg) kind); [ | discriminate ].
    destruct (bound_if_symbol statics arg); [ | discriminate ].
    intros [= <-]. exact Harg.
  - destruct (int_of_value kind arg); [ | discriminate ].
    intros [= <-]. apply P_const, const_of_int_no_region_poison.
  - destruct (int_of_value src arg); [ | discriminate ].
    intros [= <-]. apply P_const, const_of_int_no_region_poison.
  - destruct (int_of_value Standard_tagged_immediate arg) as [i | ];
      [ | discriminate ].
    destruct (i =? 0)%Z; [ intros [= <-]; apply P_const; reflexivity | ].
    destruct (i =? 1)%Z; [ | discriminate ].
    intros [= <-]. apply P_const. reflexivity.
  - destruct (boxed_number_of statics arg) as [c | ]; [ | discriminate ].
    destruct (is_number_of_kind kind c) eqn:Hkind; [ | discriminate ].
    intros [= <-]. apply P_const.
    exact (is_number_of_kind_no_region_poison kind c Hkind).
  - contradiction (Hnot kind eq_refl).
  - destruct (int_of_value Standard_tagged_immediate arg); [ | discriminate ].
    intros [= <-]. apply P_const. reflexivity.
  - destruct (int_of_value Standard_naked_immediate arg); [ | discriminate ].
    intros [= <-]. apply P_const. reflexivity.
  - destruct reinterpret; cbn [eval_reinterpret_64_bit_word];
      (destruct (int_of_value _ arg); [ | discriminate ]);
      intros [= <-]; apply P_const; reflexivity.
Qed.

Lemma eval_binary_primitive_reads (p : binary_primitive)
    (arg1 arg2 v : value) :
  P arg1 -> eval_binary_primitive statics p arg1 arg2 = Some v -> P v.
Proof.
  intros Harg1.
  destruct p as [array_kind load_kind mut | width | op | kind op | kind op
                | kind behaviour];
    cbn [eval_binary_primitive].
  - destruct (array_of statics arg1) as [elements | ] eqn:Harray;
      [ | discriminate ].
    intros Hloaded.
    exact
      (array_of_reads arg1 elements Harg1 Harray v
         (proj1 (array_loaded_element _ _ _ _ Hloaded))).
  - unfold eval_string_load.
    destruct (string_of statics arg1) as [bytes | ]; [ | discriminate ].
    destruct (int_of_value Standard_naked_nativeint arg2) as [index | ];
      [ | discriminate ].
    cbv zeta. destruct (_ && _); [ | discriminate ].
    destruct width; intros [= <-]; apply P_const; reflexivity.
  - destruct (eval_phys_equal statics arg1 arg2); [ | discriminate ].
    intros [= <-]. apply P_const. reflexivity.
  - destruct (int_of_value kind arg1) as [a | ]; [ | discriminate ].
    destruct (int_of_value kind arg2) as [b | ]; [ | discriminate ].
    destruct (eval_int_arith (int_width kind) op a b); [ | discriminate ].
    intros [= <-]. apply P_const, const_of_int_no_region_poison.
  - destruct (int_of_value kind arg1) as [a | ]; [ | discriminate ].
    destruct (int_of_value Standard_naked_immediate arg2) as [amount | ];
      [ | discriminate ].
    destruct ((0 <=? amount)%Z && (amount <? shift_bound kind)%Z);
      [ | discriminate ].
    intros [= <-]. apply P_const, const_of_int_no_region_poison.
  - destruct (int_of_value kind arg1) as [a | ]; [ | discriminate ].
    destruct (int_of_value kind arg2) as [b | ]; [ | discriminate ].
    intros [= <-]. apply P_const. reflexivity.
Qed.

(* The value slots of a closure with the property have it. *)
Lemma closure_of_reads (cv : value) (code : code_id)
    (value_slots : list (value_slot * value)) :
  P cv ->
  closure_of statics cv = Some (code, value_slots) ->
  ∀ w : value, In w (map snd value_slots) -> P w.
Proof.
  intros Hcv. destruct cv; simpl; try discriminate.
  - intros [= <- <-]. exact (fun w => P_parts _ w Hcv).
  - destruct (lookup slot function_decls); simpl; [ | discriminate ].
    intros [= <- <-]. exact (fun w => P_parts _ w Hcv).
  - destruct (lookup s (static_values statics)) as [allocated | ] eqn:Hlookup;
      [ | discriminate ].
    destruct allocated; try discriminate.
    + intros [= <- <-]. exact (proj1 (P_static s _ Hcv Hlookup)).
    + destruct (lookup slot function_decls); simpl; [ | discriminate ].
      intros [= <- <-]. exact (proj1 (P_static s _ Hcv Hlookup)).
Qed.

Lemma closure_in_set_reads (cv : value) (at_slot : function_slot)
    (function_decls : list (function_slot * code_id))
    (value_slots : list (value_slot * value)) :
  P cv ->
  closure_in_set statics cv = Some (at_slot, function_decls, value_slots) ->
  ∀ w : value, In w (map snd value_slots) -> P w.
Proof.
  intros Hcv. destruct cv; simpl; try discriminate.
  - intros [= <- <- <-]. exact (fun w => P_parts _ w Hcv).
  - destruct (lookup slot function_decls0); simpl; [ | discriminate ].
    intros [= <- <- <-]. exact (fun w => P_parts _ w Hcv).
  - destruct (lookup s (static_values statics)) as [allocated | ] eqn:Hlookup;
      [ | discriminate ].
    destruct allocated; try discriminate.
    + intros [= <- <- <-]. exact (proj1 (P_static s _ Hcv Hlookup)).
    + destruct (lookup slot function_decls0); simpl; [ | discriminate ].
      intros [= <- <- <-]. exact (proj1 (P_static s _ Hcv Hlookup)).
Qed.

Lemma project_function_slot_reads (move_from move_to : function_slot)
    (cv v : value) :
  P cv -> project_function_slot statics move_from move_to cv = Some v -> P v.
Proof.
  intros Hcv. unfold project_function_slot. cbv zeta.
  destruct cv; try discriminate.
  - destruct (Pos.eqb move_from unnamed_function_slot); [ | discriminate ].
    destruct (Pos.eqb move_to unnamed_function_slot); [ | discriminate ].
    intros [= <-]. exact Hcv.
  - destruct (Pos.eqb slot move_from); [ | discriminate ].
    destruct (lookup slot function_decls); simpl; [ | discriminate ].
    destruct (lookup move_to function_decls); simpl; [ | discriminate ].
    intros [= <-]. exact (P_move _ _ _ _ Hcv).
  - destruct (lookup s (static_values statics)) as [allocated | ] eqn:Hlookup;
      [ | discriminate ].
    destruct allocated; try discriminate.
    + destruct (Pos.eqb move_from unnamed_function_slot); [ | discriminate ].
      destruct (Pos.eqb move_to unnamed_function_slot); [ | discriminate ].
      intros [= <-]. exact Hcv.
    + destruct (Pos.eqb slot move_from); [ | discriminate ].
      destruct (lookup slot function_decls); simpl; [ | discriminate ].
      destruct (lookup move_to function_decls); simpl; [ | discriminate ].
      destruct (lookup move_to closure_symbols) as [s' | ] eqn:Hs'; simpl;
        [ | discriminate ].
      intros [= <-]. apply (proj2 (P_static s _ Hcv Hlookup)). simpl.
      apply in_map_iff. exists (move_to, s').
      split; [ reflexivity | exact (lookup_in _ _ _ Hs') ].
Qed.

Lemma eval_named_reads (vars : env) (d : named) (v : value) :
  is_allocation d = false ->
  (∀ (s : simple) (w : value),
    In s (simples_of_named d) -> eval_simple vars s = Some w -> P w) ->
  eval_named statics vars d = Some v ->
  P v.
Proof.
  intros Hnot Hsimples.
  destruct d as [s | p args | code value_slots | slot function_decls value_slots
                | slot closure | project_from slot closure
                | move_from move_to closure];
    simpl in Hnot, Hsimples |- *; try discriminate.
  - intros Hs. exact (Hsimples s v (or_introl eq_refl) Hs).
  - destruct (eval_simples vars args) as [vs | ] eqn:Hargs; simpl;
      [ | discriminate ].
    intros Hprim.
    pose proof (eval_simples_forall P vars args vs Hsimples Hargs) as Hvs.
    destruct (eval_prim_arity statics p vs v Hprim)
      as [(p' & v1 & -> & -> & Hp)
         | [(p' & v1 & v2 & -> & -> & Hp) | (p' & -> & Hp)]].
    + apply (eval_unary_primitive_reads p' v1 v);
        [ | exact (Forall_inv Hvs) | exact Hp ].
      intros kind ->. discriminate Hnot.
    + exact (eval_binary_primitive_reads p' v1 v2 v (Forall_inv Hvs) Hp).
    + destruct p'; discriminate Hnot.
  - destruct (eval_simple vars closure) as [cv | ] eqn:Hclosure; simpl;
      [ | discriminate ].
    destruct (closure_of statics cv) as [[code value_slots] | ] eqn:Hof; simpl;
      [ | discriminate ].
    intros Hlookup.
    apply
      (closure_of_reads cv code value_slots
         (Hsimples closure cv (or_introl eq_refl) Hclosure) Hof).
    apply in_map_iff. exists (slot, v).
    split; [ reflexivity | exact (lookup_in _ _ _ Hlookup) ].
  - destruct (eval_simple vars closure) as [cv | ] eqn:Hclosure; simpl;
      [ | discriminate ].
    unfold project_value_slot.
    destruct (closure_in_set statics cv) as [[[at_slot decls] value_slots] | ]
      eqn:Hin; simpl; [ | discriminate ].
    destruct (Pos.eqb at_slot project_from); [ | discriminate ].
    intros Hlookup.
    apply
      (closure_in_set_reads cv at_slot decls value_slots
         (Hsimples closure cv (or_introl eq_refl) Hclosure) Hin).
    apply in_map_iff. exists (slot, v).
    split; [ reflexivity | exact (lookup_in _ _ _ Hlookup) ].
  - destruct (eval_simple vars closure) as [cv | ] eqn:Hclosure; simpl;
      [ | discriminate ].
    exact
      (project_function_slot_reads move_from move_to cv v
         (Hsimples closure cv (or_introl eq_refl) Hclosure)).
Qed.

(* The same on a heap, of a named expression that is not a primitive
   of the heap: the property is asked of the values of the cells of the
   locations among the values of the simples too. *)
Lemma exec_named_reads (hp hp' : heap) (rd rd' : read_record) (vars : env)
    (d : named) (v : value) :
  is_allocation d = false ->
  named_changes_heap d = false ->
  (∀ (s : simple) (w : value),
    In s (simples_of_named d) -> eval_simple vars s = Some w -> P w) ->
  (∀ (s : simple) (l : location) (c : cell) (w : value),
    In s (simples_of_named d) ->
    eval_simple vars s = Some (Value_location l) ->
    lookup l hp = Some c ->
    In w (values_of_cell c) ->
    P w) ->
  exec_named statics hp rd vars d = Some (v, hp', rd') ->
  P v.
Proof.
  intros Hnot Hheap Hsimples Hcells Hexec.
  assert (Hcases :
    (∃ (p : prim) (args : list simple), d = Prim p args)
    ∨ (∀ (p : prim) (args : list simple), d <> Prim p args)).
  { destruct d;
      first [ left; eexists _, _; reflexivity | right; intros; discriminate ]. }
  destruct Hcases as [(p & args & ->) | Hnot_prim].
  - unfold exec_named in Hexec.
    destruct (eval_simples vars args) as [vs | ] eqn:Hargs;
      [ | discriminate ].
    simpl in Hexec.
    destruct (exec_prim statics hp p vs) as [[v0 hp0] | ] eqn:Hprim;
      [ | discriminate ].
    simpl in Hexec. injection Hexec as -> _ _.
    assert (Hp : ∀ q : heap_primitive, p <> Heap q).
    { intros q ->. discriminate Hheap. }
    destruct (exec_prim_reads_cases _ _ _ _ _ _ Hp Hprim)
      as [Heval
         | [(l & c & Hl & Hv & Hshape)
           | [(l & kind & -> & _ & Hvs) | (c & -> & Hc)]]].
    + apply (eval_named_reads vars (Prim p args) v Hnot Hsimples).
      simpl. rewrite Hargs. exact Heval.
    + assert (Hin : In (Value_location l) vs).
      { destruct Hshape
          as [(access & mut & field & _ & ->)
             | (kind & load_kind & mut & v2 & _ & ->)];
          left; reflexivity. }
      destruct (eval_simples_in vars args vs _ Hargs Hin) as (s & Hs & Hw).
      exact (Hcells s l c v Hs Hw Hl Hv).
    + assert (Hin : In (Value_location l) vs)
        by (rewrite Hvs; left; reflexivity).
      destruct (eval_simples_in vars args vs _ Hargs Hin) as (s & Hs & Hw).
      exact (Hsimples s _ Hs Hw).
    + exact (P_const c Hc).
  - rewrite (exec_named_not_prim statics hp rd vars d Hnot_prim) in Hexec.
    destruct (eval_named statics vars d) as [v0 | ] eqn:Hnamed;
      [ | discriminate ].
    simpl in Hexec. injection Hexec as -> _ _.
    exact (eval_named_reads vars d v Hnot Hsimples Hnamed).
Qed.

End Reads.

(* What a named expression leaves in the heap: a property that holds of
   the values of the heap and of the values of the simples holds of the
   values of the heap after. *)
Lemma exec_named_cells (Q : value -> Prop) (statics : static_env)
    (hp hp' : heap) (rd rd' : read_record) (vars : env) (d : named)
    (v : value) :
  heap_values_all Q hp ->
  (∀ (s : simple) (w : value),
    In s (simples_of_named d) -> eval_simple vars s = Some w -> Q w) ->
  exec_named statics hp rd vars d = Some (v, hp', rd') ->
  heap_values_all Q hp'.
Proof.
  intros Hcells Hsimples Hexec.
  assert (Hcases :
    (∃ (p : prim) (args : list simple), d = Prim p args)
    ∨ (∀ (p : prim) (args : list simple), d <> Prim p args)).
  { destruct d;
      first [ left; eexists _, _; reflexivity | right; intros; discriminate ]. }
  destruct Hcases as [(p & args & ->) | Hnot_prim].
  - unfold exec_named in Hexec.
    destruct (eval_simples vars args) as [vs | ] eqn:Hargs;
      [ | discriminate ].
    simpl in Hexec.
    destruct (exec_prim statics hp p vs) as [[v0 hp0] | ] eqn:Hprim;
      [ | discriminate ].
    simpl in Hexec. injection Hexec as _ <- _.
    apply (exec_prim_heap_cells Q statics hp hp0 p vs v0 Hcells);
      [ | exact Hprim ].
    intros arg Hin.
    destruct (eval_simples_in vars args vs _ Hargs Hin) as (s & Hs & Hw).
    exact (Hsimples s arg Hs Hw).
  - rewrite (exec_named_not_prim statics hp rd vars d Hnot_prim) in Hexec.
    destruct (eval_named statics vars d) as [v0 | ]; [ | discriminate ].
    simpl in Hexec. injection Hexec as _ <- _. exact Hcells.
Qed.

(* What a primitive of the heap gives, as a named expression. *)
Lemma exec_named_heap_value (statics : static_env) (hp hp' : heap)
    (rd rd' : read_record) (vars : env) (d : named) (v : value) :
  named_changes_heap d = true ->
  exec_named statics hp rd vars d = Some (v, hp', rd') ->
  v = Value_location (fresh_location hp) ∨ v = unit_value.
Proof.
  intros Hheap Hexec.
  destruct d as [ | p args | | | | | ]; try discriminate Hheap.
  destruct p as [ | | | q | ]; try discriminate Hheap.
  unfold exec_named in Hexec.
  destruct (eval_simples vars args) as [vs | ]; [ | discriminate ].
  simpl in Hexec.
  destruct (exec_prim statics hp (Heap q) vs) as [[v0 hp0] | ] eqn:Hprim;
    [ | discriminate ].
  simpl in Hexec. injection Hexec as -> _ _.
  exact (exec_prim_heap_value _ _ _ _ _ _ Hprim).
Qed.

(* A named expression that allocates what is no cell is run by
   [eval_named], and leaves the heap. *)
Lemma exec_named_makes (statics : static_env) (hp hp' : heap)
    (rd rd' : read_record) (vars : env) (d : named) (v : value) :
  is_allocation d = true ->
  exec_named statics hp rd vars d = Some (v, hp', rd') ->
  eval_named statics vars d = Some v ∧ hp' = hp.
Proof.
  intros His Hexec.
  assert (Hcases :
    (∃ (p : prim) (args : list simple), d = Prim p args)
    ∨ (∀ (p : prim) (args : list simple), d <> Prim p args)).
  { destruct d;
      first [ left; eexists _, _; reflexivity | right; intros; discriminate ]. }
  destruct Hcases as [(p & args & ->) | Hnot_prim].
  - assert (Hreads : prim_reads_heap p = false).
    { destruct p as [u | b | variadic | q | rp]; try discriminate His;
        try reflexivity.
      destruct u; try discriminate His. reflexivity. }
    destruct (exec_named_eval_named _ _ _ _ _ _ _ _ _ Hreads Hexec)
      as (Hnamed & -> & _).
    split; [ exact Hnamed | reflexivity ].
  - rewrite (exec_named_not_prim statics hp rd vars d Hnot_prim) in Hexec.
    destruct (eval_named statics vars d) as [v0 | ]; [ | discriminate ].
    simpl in Hexec. injection Hexec as -> <- _. split; reflexivity.
Qed.

(* [heap_values_all] of a test: the test of every value of every entry. *)
Lemma heap_values_all_forallb (f : value -> bool) (hp : heap) :
  forallb (fun '(_, c) => forallb f (values_of_cell c)) hp = true
  <-> heap_values_all (fun v => f v = true) hp.
Proof.
  unfold heap_values_all. rewrite forallb_forall. split.
  - intros H l c v Hin Hv. specialize (H (l, c) Hin). cbv beta iota in H.
    exact (forallb_in _ _ v H Hv).
  - intros H [l c] Hin. apply forallb_forall. intros v Hv.
    exact (H l c v Hin Hv).
Qed.

(* What an allocation makes holds directly the values of its simples. *)
Lemma eval_named_makes (statics : static_env) (vars : env) (d : named)
    (v : value) :
  is_allocation d = true ->
  eval_named statics vars d = Some v ->
  is_made v = true
  ∧ ∀ w : value,
      In w (value_parts v) ->
      ∃ s : simple, In s (simples_of_named d) ∧ eval_simple vars s = Some w.
Proof.
  intros His.
  destruct d as [s | p args | code value_slots | slot function_decls value_slots
                | slot closure | project_from slot closure
                | move_from move_to closure];
    simpl in His |- *; try discriminate.
  - destruct (eval_simples vars args) as [vs | ] eqn:Hargs; simpl;
      [ | discriminate ].
    intros Hprim.
    destruct (eval_prim_arity statics p vs v Hprim)
      as [(p' & v1 & -> & -> & Hp)
         | [(p' & v1 & v2 & -> & -> & Hp) | (p' & -> & Hp)]].
    + destruct p'; try discriminate His. simpl in Hp.
      destruct v1; try discriminate.
      destruct (is_number_of_kind kind c); [ | discriminate ].
      injection Hp as <-. split; [ reflexivity | ].
      intros w [<- | []].
      exact (eval_simples_in vars args _ _ Hargs (or_introl eq_refl)).
    + discriminate His.
    + destruct p' as [[tag] | kind]; simpl in Hp.
      * destruct
          (forallb (fun v => kind_eqb (kind_of_value v) Value) vs
           && forallb (bound_if_symbol statics) vs);
          [ | discriminate ].
        injection Hp as <-. split; [ reflexivity | ].
        exact (fun w => eval_simples_in vars args vs w Hargs).
      * destruct (array_made kind vs) as [elements | ] eqn:Hmade; simpl in Hp;
          [ | discriminate ].
        destruct (forallb (bound_if_symbol statics) vs); [ | discriminate ].
        injection Hp as <-.
        destruct (array_made_of_arguments kind vs elements Hmade) as [-> _].
        split; [ reflexivity | ].
        exact (fun w => eval_simples_in vars args vs w Hargs).
  - destruct (eval_value_slots vars value_slots) as [vs | ] eqn:Hslots; simpl;
      [ | discriminate ].
    intros [= <-]. split; [ reflexivity | ].
    exact (fun w => eval_value_slots_in vars value_slots vs w Hslots).
  - destruct (lookup slot function_decls); simpl; [ | discriminate ].
    destruct (eval_value_slots vars value_slots) as [vs | ] eqn:Hslots; simpl;
      [ | discriminate ].
    intros [= <-]. split; [ reflexivity | ].
    exact (fun w => eval_value_slots_in vars value_slots vs w Hslots).
Qed.

(* What a static constant is bound to holds directly the values of its
   simples, and the symbols that the constant gives a set. *)
Lemma eval_static_const_makes (vars : env) (const : static_const)
    (allocated : statically_allocated_value) :
  eval_static_const vars const = Some allocated ->
  (∀ w : value,
    In w (allocated_values allocated) ->
    ∃ s : simple,
      In s (simples_of_static_const const) ∧ eval_simple vars s = Some w)
  ∧ allocated_symbols allocated
    = match const with
      | Static_const_closure_of_set _ closure_symbols _ _ =>
        map snd closure_symbols
      | _ => []
      end.
Proof.
  destruct const
    as [code value_slots | slot closure_symbols function_decls value_slots
       | tag fields | contents | bytes | elements];
    simpl.
  - destruct (eval_value_slots vars value_slots) as [vs | ] eqn:Hslots; simpl;
      [ | discriminate ].
    intros [= <-]. split; [ | reflexivity ].
    exact (fun w => eval_value_slots_in vars value_slots vs w Hslots).
  - destruct (lookup slot function_decls); simpl; [ | discriminate ].
    destruct (eval_value_slots vars value_slots) as [vs | ] eqn:Hslots; simpl;
      [ | discriminate ].
    intros [= <-]. split; [ | reflexivity ].
    exact (fun w => eval_value_slots_in vars value_slots vs w Hslots).
  - destruct (eval_simples vars fields) as [vs | ] eqn:Hfields; simpl;
      [ | discriminate ].
    intros [= <-]. split; [ | reflexivity ].
    exact (fun w => eval_simples_in vars fields vs w Hfields).
  - destruct (eval_simple vars contents) as [v | ] eqn:Hcontents; simpl;
      [ | discriminate ].
    intros [= <-]. split; [ | reflexivity ].
    intros w [<- | []]. exists contents.
    split; [ left; reflexivity | exact Hcontents ].
  - intros [= <-]. split; [ intros w [] | reflexivity ].
  - destruct (eval_simples vars elements) as [vs | ] eqn:Helements; simpl;
      [ | discriminate ].
    intros [= <-]. split; [ | reflexivity ].
    exact (fun w => eval_simples_in vars elements vs w Helements).
Qed.

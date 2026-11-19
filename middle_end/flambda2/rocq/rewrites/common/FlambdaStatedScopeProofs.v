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

(* Proofs of the statements of FlambdaStatedScopeSpec.v. There is no
   statement here, and nothing here needs review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaValueSpec FlambdaPrimitiveSpec FlambdaSubkindSpec
  FlambdaSubkindOrderSpec FlambdaHeapSpec FlambdaSemanticsSpec
  FlambdaWellFormedSpec FlambdaSubkindBinderSpec FlambdaStatedScopeSpec.
From OxCamlFlambda2 Require FlambdaPrimitiveProofs FlambdaSubkindProofs
  FlambdaSubkindOrderProofs FlambdaSubkindBinderProofs FlambdaHeapProofs
  FlambdaSemanticsProofs.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* The test of the machine, of one value                                    *)
(* ======================================================================== *)

Theorem value_at_any_value : value_at_any_value_statement.
Proof. exact FlambdaSubkindProofs.value_at_any_value. Qed.

Theorem value_at_const : value_at_const_statement.
Proof.
  intros statics ps c. rewrite !FlambdaSubkindProofs.value_at_eq.
  apply FlambdaSubkindProofs.subkind_at_const.
Qed.

Theorem value_at_weaken : value_at_weaken_statement.
Proof. exact FlambdaSubkindOrderProofs.value_at_weaken. Qed.

Theorem value_at_extend : value_at_extend_statement.
Proof.
  intros statics statics' ps v [_ Hvalues] Hv.
  pose proof
    (FlambdaSubkindProofs.values_of_subkinds_extend statics statics' [v] [ps]
       Hvalues) as H.
  cbn [values_of_subkinds] in H. rewrite !andb_true_r in H. exact (H Hv).
Qed.

(* A tagged immediate is of the subkind of tagged immediates. *)
Lemma tagged_value_at (statics : static_env) (v : value) :
  is_tagged_immediate v = true -> value_at statics tagged v = true.
Proof.
  intros Hv. destruct v as [c | | | | | | | | ]; try discriminate Hv.
  destruct c; try discriminate Hv. exact Hv.
Qed.

Lemma tagged_const_at (statics : static_env) (i : Z) :
  in_range (int_width Standard_tagged_immediate) i = true ->
  value_at statics tagged (Value_const (Tagged_immediate i)) = true.
Proof.
  intros Hi. apply tagged_value_at. unfold is_tagged_immediate.
  cbn [int_of_value]. rewrite Hi. reflexivity.
Qed.

(* ======================================================================== *)
(* The machine keeps what is known                                          *)
(* ======================================================================== *)

Theorem stated_hold_extend : stated_hold_extend_statement.
Proof.
  intros statics statics' S vars Hextend Hhold x ps v HS Hv.
  exact (value_at_extend statics statics' ps v Hextend (Hhold x ps v HS Hv)).
Qed.

Lemma stated_hold_nil (statics : static_env) (vars : env) :
  stated_hold statics [] vars.
Proof. intros x ps v HS. discriminate HS. Qed.

Lemma stated_hold_cons (statics : static_env) (S : stated_in_scope)
    (vars : env) (x : variable) (ps : param_subkind) (v : value) :
  stated_hold statics S vars ->
  value_at statics ps v = true ->
  stated_hold statics ((x, ps) :: S) ((x, v) :: vars).
Proof.
  intros Hhold Hv y ps' w. simpl.
  destruct (Pos.eqb y x); [ intros [= <-] [= <-]; exact Hv | apply Hhold ].
Qed.

Theorem stated_of_simple_sound : stated_of_simple_sound_statement.
Proof.
  intros statics S vars s v Hhold Heval.
  destruct s as [x | sym | c]; cbn [stated_of_simple eval_simple] in *.
  - unfold stated_of_var. destruct (lookup x S) as [ps | ] eqn:HS.
    + exact (Hhold x ps v HS Heval).
    + apply value_at_any_value.
  - injection Heval as <-. reflexivity.
  - injection Heval as <-.
    destruct c as [ | | | | | | | | | | | | | | ];
      try apply value_at_any_value.
    destruct (is_tagged_immediate (Value_const (Tagged_immediate i)))
      eqn:Hrange; [ | apply value_at_any_value ].
    unfold value_at. cbn [fst snd value_of_subkind existsb].
    rewrite Hrange, Z.eqb_refl. reflexivity.
Qed.

(* ---- What a primitive makes -------------------------------------------- *)

(* A primitive that has a value on a heap, by cases: one of the heap; a
   unary one that reads the cell of a location; a binary one, other
   than [Phys_equal], that reads the cell of its first argument; and
   otherwise FlambdaPrimitiveSpec.eval_prim on the arguments. *)
Lemma exec_prim_unary_cases (statics : static_env) (h h' : heap)
    (p : unary_primitive) (vs : list value) (v : value) :
  exec_prim statics h (Unary p) vs = Some (v, h') ->
  ∃ v1 : value,
    vs = [v1]
    ∧ ((∃ (l : location) (c : cell),
         v1 = Value_location l ∧ lookup l h = Some c
         ∧ exec_unary_on_cell p l c = Some v)
       ∨ eval_unary_primitive statics p v1 = Some v).
Proof.
  unfold exec_prim.
  destruct (forallb (argument_bound statics h) vs); [ | discriminate ].
  rewrite FlambdaHeapProofs.exec_bound_prim_unary_form.
  assert (Hpure :
    (let* result := eval_prim statics (Unary p) vs in Some (result, h))
    = Some (v, h') ->
    ∃ v1 : value, vs = [v1] ∧ eval_unary_primitive statics p v1 = Some v).
  { destruct (eval_prim statics (Unary p) vs) as [w | ] eqn:Heval;
      [ | discriminate ].
    cbn. intros [= <- _].
    destruct (FlambdaPrimitiveProofs.eval_prim_arity statics _ vs w Heval)
      as [(p' & v1 & [= <-] & -> & Hp)
         | [(p' & v1 & v2 & Habs & _) | (p' & Habs & _)]];
      try discriminate Habs.
    exists v1. split; [ reflexivity | exact Hp ]. }
  destruct vs as [ | v1 [ | v2 vs]];
    try (intros H; destruct (Hpure H) as (w & Hw & _); discriminate Hw).
  destruct (FlambdaHeapProofs.location_of v1) as [l | ] eqn:Hl.
  - apply FlambdaHeapProofs.location_of_some in Hl. subst v1.
    destruct (lookup l h) as [c | ] eqn:Hc; [ | discriminate ]. cbn.
    destruct (exec_unary_on_cell p l c) as [w | ] eqn:Hw; [ | discriminate ].
    cbn. intros [= <- _]. exists (Value_location l).
    split; [ reflexivity | ]. left. exists l, c. auto.
  - intros H. destruct (Hpure H) as (w & [= <-] & Hw).
    exists v1. split; [ reflexivity | right; exact Hw ].
Qed.

Lemma eval_prim_unary_some (statics : static_env) (p : unary_primitive)
    (vs : list value) (v : value) :
  eval_prim statics (Unary p) vs = Some v ->
  ∃ v1 : value, vs = [v1] ∧ eval_unary_primitive statics p v1 = Some v.
Proof.
  intros Heval.
  destruct (FlambdaPrimitiveProofs.eval_prim_arity statics _ vs v Heval)
    as [(p' & v1 & [= <-] & -> & Hp)
       | [(p' & v1 & v2 & Habs & _) | (p' & Habs & _)]];
    try discriminate Habs.
  exists v1. auto.
Qed.

Lemma eval_prim_binary_some (statics : static_env) (p : binary_primitive)
    (vs : list value) (v : value) :
  eval_prim statics (Binary p) vs = Some v ->
  ∃ v1 v2 : value,
    vs = [v1; v2] ∧ eval_binary_primitive statics p v1 v2 = Some v.
Proof.
  intros Heval.
  destruct (FlambdaPrimitiveProofs.eval_prim_arity statics _ vs v Heval)
    as [(p' & v1 & Habs & _)
       | [(p' & v1 & v2 & [= <-] & -> & Hp) | (p' & Habs & _)]];
    try discriminate Habs.
  exists v1, v2. auto.
Qed.

(* A primitive that does not read the heap, on a heap. *)
Lemma exec_pure_unary (statics : static_env) (h h' : heap)
    (p : unary_primitive) (vs : list value) (v : value) :
  prim_reads_heap (Unary p) = false ->
  exec_prim statics h (Unary p) vs = Some (v, h') ->
  ∃ v1 : value, vs = [v1] ∧ eval_unary_primitive statics p v1 = Some v.
Proof.
  intros Hreads Hexec.
  destruct
    (FlambdaHeapProofs.exec_prim_eval_prim statics h h' _ vs v Hreads Hexec)
    as [Heval _].
  exact (eval_prim_unary_some statics p vs v Heval).
Qed.

Lemma exec_pure_binary (statics : static_env) (h h' : heap)
    (p : binary_primitive) (vs : list value) (v : value) :
  prim_reads_heap (Binary p) = false ->
  exec_prim statics h (Binary p) vs = Some (v, h') ->
  ∃ v1 v2 : value,
    vs = [v1; v2] ∧ eval_binary_primitive statics p v1 v2 = Some v.
Proof.
  intros Hreads Hexec.
  destruct
    (FlambdaHeapProofs.exec_prim_eval_prim statics h h' _ vs v Hreads Hexec)
    as [Heval _].
  exact (eval_prim_binary_some statics p vs v Heval).
Qed.

Lemma exec_pure_variadic (statics : static_env) (h h' : heap)
    (p : variadic_primitive) (vs : list value) (v : value) :
  exec_prim statics h (Variadic p) vs = Some (v, h') ->
  eval_variadic_primitive statics p vs = Some v.
Proof.
  intros Hexec.
  destruct
    (FlambdaHeapProofs.exec_prim_eval_prim statics h h' (Variadic p) vs v
       eq_refl Hexec)
    as [Heval _].
  rewrite FlambdaPrimitiveProofs.eval_prim_variadic in Heval. exact Heval.
Qed.

(* The integer that an integer of the kind of tagged immediates is made
   of, where it is in range, is a tagged immediate. *)
Lemma tagged_int_at (statics : static_env) (i : Z) :
  in_range (int_width Standard_tagged_immediate) i = true ->
  value_at statics tagged
    (Value_const (const_of_int Standard_tagged_immediate i))
  = true.
Proof. exact (tagged_const_at statics i). Qed.

Lemma one_le_tagged_width : 1 <= int_width Standard_tagged_immediate.
Proof. cbn. lia. Qed.

(* The unary primitives that make a tagged immediate. *)
Lemma tag_immediate_tagged (statics : static_env) (h h' : heap)
    (vs : list value) (v : value) :
  exec_prim statics h (Unary Tag_immediate) vs = Some (v, h') ->
  value_at statics tagged v = true.
Proof.
  intros Hexec.
  destruct (exec_pure_unary statics h h' Tag_immediate vs v eq_refl Hexec)
    as (v1 & _ & Hv).
  cbn [eval_unary_primitive] in Hv.
  destruct (int_of_value Standard_naked_immediate v1) as [i | ] eqn:Hint;
    [ | discriminate Hv ].
  cbv beta iota in Hv. injection Hv as <-.
  apply FlambdaPrimitiveProofs.int_of_value_const in Hint as [_ Hrange].
  exact (tagged_const_at statics i Hrange).
Qed.

Lemma boolean_not_tagged (statics : static_env) (h h' : heap)
    (vs : list value) (v : value) :
  exec_prim statics h (Unary Boolean_not) vs = Some (v, h') ->
  value_at statics tagged v = true.
Proof.
  intros Hexec.
  destruct (exec_pure_unary statics h h' Boolean_not vs v eq_refl Hexec)
    as (v1 & _ & Hv).
  cbn [eval_unary_primitive] in Hv.
  destruct (int_of_value Standard_tagged_immediate v1) as [i | ];
    [ | discriminate Hv ].
  cbv beta iota in Hv.
  destruct (i =? 0); [ injection Hv as <-; reflexivity | ].
  destruct (i =? 1); [ injection Hv as <-; reflexivity | discriminate Hv ].
Qed.

Lemma unary_int_arith_tagged (statics : static_env) (h h' : heap)
    (op : unary_int_arith_op) (vs : list value) (v : value) :
  exec_prim statics h (Unary (Unary_int_arith Standard_tagged_immediate op))
    vs
  = Some (v, h') ->
  value_at statics tagged v = true.
Proof.
  intros Hexec.
  destruct
    (exec_pure_unary statics h h'
       (Unary_int_arith Standard_tagged_immediate op) vs v eq_refl Hexec)
    as (v1 & _ & Hv).
  cbn [eval_unary_primitive] in Hv.
  destruct (int_of_value Standard_tagged_immediate v1) as [i | ];
    [ | discriminate Hv ].
  cbv beta iota in Hv. injection Hv as <-.
  apply tagged_int_at.
  apply FlambdaPrimitiveProofs.eval_unary_int_arith_in_range.
Qed.

Lemma num_conv_tagged (statics : static_env) (h h' : heap)
    (src : standard_int) (vs : list value) (v : value) :
  exec_prim statics h (Unary (Num_conv src Standard_tagged_immediate)) vs
  = Some (v, h') ->
  value_at statics tagged v = true.
Proof.
  intros Hexec.
  destruct
    (exec_pure_unary statics h h' (Num_conv src Standard_tagged_immediate)
       vs v eq_refl Hexec)
    as (v1 & _ & Hv).
  cbn [eval_unary_primitive] in Hv.
  destruct (int_of_value src v1) as [i | ]; [ | discriminate Hv ].
  cbv beta iota in Hv. injection Hv as <-.
  apply tagged_int_at.
  apply FlambdaPrimitiveProofs.wrap_in_range, one_le_tagged_width.
Qed.

(* The binary primitives that make a tagged immediate. *)
Lemma int_arith_tagged (statics : static_env) (h h' : heap)
    (op : binary_int_arith_op) (vs : list value) (v : value) :
  exec_prim statics h (Binary (Int_arith Standard_tagged_immediate op)) vs
  = Some (v, h') ->
  value_at statics tagged v = true.
Proof.
  intros Hexec.
  destruct
    (exec_pure_binary statics h h' (Int_arith Standard_tagged_immediate op)
       vs v eq_refl Hexec)
    as (v1 & v2 & _ & Hv).
  cbn [eval_binary_primitive] in Hv.
  destruct (int_of_value Standard_tagged_immediate v1) as [a | ];
    [ | discriminate Hv ].
  destruct (int_of_value Standard_tagged_immediate v2) as [b | ];
    [ | discriminate Hv ].
  cbv beta iota in Hv.
  destruct (eval_int_arith (int_width Standard_tagged_immediate) op a b)
    as [result | ] eqn:Hresult; [ | discriminate Hv ].
  cbv beta iota in Hv. injection Hv as <-.
  apply tagged_int_at.
  exact
    (FlambdaPrimitiveProofs.eval_int_arith_in_range _ op a b result
       one_le_tagged_width Hresult).
Qed.

Lemma int_shift_tagged (statics : static_env) (h h' : heap)
    (op : int_shift_op) (vs : list value) (v : value) :
  exec_prim statics h (Binary (Int_shift Standard_tagged_immediate op)) vs
  = Some (v, h') ->
  value_at statics tagged v = true.
Proof.
  intros Hexec.
  destruct
    (exec_pure_binary statics h h' (Int_shift Standard_tagged_immediate op)
       vs v eq_refl Hexec)
    as (v1 & v2 & _ & Hv).
  cbn [eval_binary_primitive] in Hv.
  destruct (int_of_value Standard_tagged_immediate v1) as [a | ];
    [ | discriminate Hv ].
  destruct (int_of_value Standard_naked_immediate v2) as [amount | ];
    [ | discriminate Hv ].
  cbv beta iota in Hv.
  destruct
    ((0 <=? amount) && (amount <? shift_bound Standard_tagged_immediate));
    [ | discriminate Hv ].
  injection Hv as <-.
  apply tagged_int_at.
  apply FlambdaPrimitiveProofs.eval_int_shift_in_range, one_le_tagged_width.
Qed.

(* [Box_number] makes the boxed number of its kind. *)
Lemma box_number_boxed (statics : static_env) (h h' : heap)
    (n : boxable_number) (vs : list value) (v : value) :
  exec_prim statics h (Unary (Box_number n)) vs = Some (v, h') ->
  value_at statics (false, Subkind_boxed_number n) v = true.
Proof.
  intros Hexec.
  destruct (exec_pure_unary statics h h' (Box_number n) vs v eq_refl Hexec)
    as (v1 & _ & Hv).
  cbn [eval_unary_primitive] in Hv.
  destruct v1 as [c | | | | | | | | ]; try discriminate Hv.
  destruct (is_number_of_kind n c) eqn:Hkind; [ | discriminate Hv ].
  injection Hv as <-.
  destruct n, c; try discriminate Hkind; reflexivity.
Qed.

(* The length of an array is a tagged immediate. *)
Lemma array_length_of_tagged (statics : static_env) (elements : list value)
    (v : value) :
  array_length_of elements = Some v -> value_at statics tagged v = true.
Proof.
  unfold array_length_of.
  destruct
    (in_range (int_width Standard_tagged_immediate)
       (Z.of_nat (length elements))) eqn:Hrange; [ | discriminate ].
  intros [= <-]. exact (tagged_const_at statics _ Hrange).
Qed.

Lemma array_length_tagged (statics : static_env) (h h' : heap)
    (kind : array_kind_for_length) (vs : list value) (v : value) :
  exec_prim statics h (Unary (Array_length kind)) vs = Some (v, h') ->
  value_at statics tagged v = true.
Proof.
  intros Hexec.
  destruct (exec_prim_unary_cases statics h h' _ vs v Hexec)
    as (v1 & _ & [(l & c & _ & _ & Hcell) | Hv]).
  - destruct c as [mut tag fields | akind elements];
      [ discriminate Hcell | ].
    exact (array_length_of_tagged statics elements v Hcell).
  - cbn [eval_unary_primitive] in Hv.
    destruct (array_of statics v1) as [elements | ]; [ | discriminate Hv ].
    exact (array_length_of_tagged statics elements v Hv).
Qed.

(* A load of an array at the load kind of immediates gives a tagged
   immediate. *)
Lemma array_loaded_immediate (statics : static_env) (elements : list value)
    (index v : value) :
  array_loaded Load_immediates elements index = Some v ->
  value_at statics tagged v = true.
Proof.
  intros Hload. apply tagged_value_at.
  exact
    (proj2
       (FlambdaPrimitiveProofs.array_loaded_element Load_immediates elements
          index v Hload)).
Qed.

Lemma array_load_immediates_tagged (statics : static_env) (h h' : heap)
    (kind : array_kind) (mut : mutability) (vs : list value) (v : value) :
  exec_prim statics h (Binary (Array_load kind Load_immediates mut)) vs
  = Some (v, h') ->
  value_at statics tagged v = true.
Proof.
  unfold exec_prim.
  destruct (forallb (argument_bound statics h) vs); [ | discriminate ].
  rewrite FlambdaHeapProofs.exec_bound_prim_binary.
  cbn [FlambdaHeapProofs.phys_equal_on_location].
  rewrite FlambdaHeapProofs.binary_on_cell_form.
  assert (Hpure :
    (let* w := eval_prim statics (Binary (Array_load kind Load_immediates mut))
                 vs in
     Some (w, h))
    = Some (v, h') ->
    value_at statics tagged v = true).
  { destruct
      (eval_prim statics (Binary (Array_load kind Load_immediates mut)) vs)
      as [w | ] eqn:Heval; [ | discriminate ].
    cbv beta iota. intros [= <- _].
    destruct (eval_prim_binary_some statics _ vs w Heval)
      as (v1 & v2 & _ & Hw).
    cbn [eval_binary_primitive] in Hw.
    destruct (array_of statics v1) as [elements | ]; [ | discriminate Hw ].
    exact (array_loaded_immediate statics elements v2 w Hw). }
  destruct vs as [ | v1 [ | v2 [ | v3 vs]]]; try exact Hpure.
  destruct (FlambdaHeapProofs.location_of v1) as [l | ]; [ | exact Hpure ].
  destruct (lookup l h) as [c | ]; [ | discriminate ]. cbv beta iota.
  destruct c as [cmut tag fields | akind elements]; [ discriminate | ].
  cbn [exec_binary_on_cell].
  destruct (load_allowed mut Cell_mutable); [ | discriminate ].
  destruct (array_loaded Load_immediates elements v2) as [w | ] eqn:Hw;
    [ | discriminate ].
  cbv beta iota. intros [= <- _].
  exact (array_loaded_immediate statics elements v2 w Hw).
Qed.

(* [Opaque_identity] gives back its argument. *)
Lemma opaque_identity_same (statics : static_env) (h h' : heap)
    (kind : flambda_kind) (vs : list value) (v : value) :
  exec_prim statics h (Unary (Opaque_identity kind)) vs = Some (v, h') ->
  vs = [v].
Proof.
  intros Hexec.
  destruct (exec_prim_unary_cases statics h h' _ vs v Hexec)
    as (v1 & -> & [(l & c & -> & _ & Hcell) | Hv]).
  - destruct c as [mut tag fields | akind elements];
      cbn [exec_unary_on_cell] in Hcell;
      (destruct (kind_eqb kind Value); [ | discriminate Hcell ]);
      injection Hcell as <-; reflexivity.
  - cbn [eval_unary_primitive] in Hv.
    destruct (kind_eqb (kind_of_value v1) kind && bound_if_symbol statics v1);
      [ | discriminate Hv ].
    injection Hv as <-. reflexivity.
Qed.

(* A load whose access kind says that the field is an immediate gives a
   tagged immediate. *)
Lemma block_load_immediate (statics : static_env) (h h' : heap)
    (tag size : option Z) (mut : mutability) (field : Z) (vs : list value)
    (v : value) :
  exec_prim statics h
    (Unary (Block_load (Block_access_values tag size Immediate) mut field)) vs
  = Some (v, h') ->
  is_tagged_immediate v = true.
Proof.
  intros Hexec.
  destruct (exec_prim_unary_cases statics h h' _ vs v Hexec)
    as (v1 & _ & [(l & c & _ & _ & Hcell) | Hv]).
  - destruct c as [cmut ctag fields | akind elements];
      [ | discriminate Hcell ].
    cbn [exec_unary_on_cell] in Hcell.
    destruct (load_allowed mut cmut); [ | discriminate Hcell ].
    unfold load_field in Hcell.
    destruct
      (known_is tag ctag && known_is size (Z.of_nat (length fields))
       && (0 <=? field));
      [ | discriminate Hcell ].
    destruct (nth_error fields (Z.to_nat field)) as [loaded | ];
      [ | discriminate Hcell ].
    cbv beta iota in Hcell.
    destruct (is_tagged_immediate loaded) eqn:Hloaded;
      [ | discriminate Hcell ].
    injection Hcell as <-. exact Hloaded.
  - cbn [eval_unary_primitive] in Hv. unfold eval_block_load in Hv.
    destruct (block_of statics v1) as [[btag fields] | ];
      [ | discriminate Hv ].
    cbv beta iota in Hv.
    destruct
      (known_is tag btag && known_is size (Z.of_nat (length fields))
       && (0 <=? field));
      [ | discriminate Hv ].
    destruct (nth_error fields (Z.to_nat field)) as [loaded | ];
      [ | discriminate Hv ].
    cbv beta iota in Hv.
    destruct (is_tagged_immediate loaded) eqn:Hloaded; [ | discriminate Hv ].
    injection Hv as <-. exact Hloaded.
Qed.

(* A load from a value known to be a block of one shape gives what the
   shape says of the field. A value that is null, a constant
   constructor or a location has no field to load. *)
Lemma block_load_of_shape (statics : static_env) (h h' : heap)
    (kind : block_access_kind) (mut : mutability) (field : Z) (v1 v : value)
    (ps : param_subkind) :
  exec_prim statics h (Unary (Block_load kind mut field)) [v1] = Some (v, h') ->
  value_at statics ps v1 = true ->
  value_at statics (stated_of_field ps field) v = true.
Proof.
  intros Hexec Hps. unfold stated_of_field.
  destruct ps as [nullable sk]. cbn [snd].
  destruct sk as [ | | n | consts shapes]; try apply value_at_any_value.
  destruct shapes as [ | tag' rest | tag' fields' rest];
    try apply value_at_any_value.
  destruct rest; try apply value_at_any_value.
  destruct (Z.leb 0 field); [ | apply value_at_any_value ].
  destruct (field_subkind_at fields' (Z.to_nat field)) as [ps' | ] eqn:Hat;
    [ | apply value_at_any_value ].
  destruct (exec_prim_unary_cases statics h h' _ [v1] v Hexec)
    as (w & [= <-] & [(l & c & -> & _ & _) | Hv]).
  - discriminate Hps.
  - cbn [eval_unary_primitive] in Hv. destruct kind as [tag size field_kind].
    unfold eval_block_load in Hv.
    destruct (block_of statics v1) as [[btag fields] | ] eqn:Hblock;
      [ | discriminate Hv ].
    cbv beta iota in Hv.
    destruct
      (known_is tag btag && known_is size (Z.of_nat (length fields))
       && (0 <=? field));
      [ | discriminate Hv ].
    destruct (nth_error fields (Z.to_nat field)) as [loaded | ] eqn:Hnth;
      [ | discriminate Hv ].
    cbv beta iota in Hv.
    assert (Hloaded : loaded = v).
    { destruct
        (match field_kind with
         | Any_value => kind_eqb (kind_of_value loaded) Value
         | Immediate => is_tagged_immediate loaded
         end);
        [ injection Hv as <-; reflexivity | discriminate Hv ]. }
    subst loaded.
    assert (Hshape :
      block_of_shapes statics (Shape_determined tag' fields' Shapes_nil) btag
        fields
      = true).
    { destruct v1 as [c | | | | | | | | ]; try discriminate Hblock;
        unfold value_at in Hps; cbn [fst snd value_of_subkind] in Hps;
        rewrite Hblock in Hps; exact Hps. }
    cbn [block_of_shapes] in Hshape.
    destruct (Z.eqb btag tag'); [ | discriminate Hshape ].
    destruct
      (FlambdaSubkindBinderProofs.fields_of_subkinds_nth statics fields'
         fields (Z.to_nat field) ps' Hshape Hat)
      as (loaded & Hnth' & _ & Hloaded).
    rewrite Hnth in Hnth'. injection Hnth' as <-. exact Hloaded.
Qed.

(* The fields of a block made of these arguments are of what is known of
   them. *)
Lemma fields_stated_sound (statics : static_env) (S : stated_in_scope)
    (vars : env) :
  stated_hold statics S vars ->
  ∀ (args : list simple) (vs : list value),
    eval_simples vars args = Some vs ->
    forallb (fun v => kind_eqb (kind_of_value v) Value) vs = true ->
    fields_of_subkinds statics (fields_stated S args) vs = true.
Proof.
  intros Hhold. induction args as [ | s args IH]; intros vs; cbn [eval_simples].
  - intros [= <-] _. reflexivity.
  - destruct (eval_simple vars s) as [v | ] eqn:Hs; [ | discriminate ].
    destruct (eval_simples vars args) as [vs' | ]; [ | discriminate ].
    cbv beta iota. intros [= <-]. cbn [forallb fields_stated]. intros Hkinds.
    apply andb_true_iff in Hkinds as [Hkind Hkinds].
    rewrite FlambdaSubkindProofs.fields_of_subkinds_cons.
    rewrite <- FlambdaSubkindProofs.value_at_eq.
    rewrite Hkind, (stated_of_simple_sound statics S vars s v Hhold Hs).
    exact (IH vs' eq_refl Hkinds).
Qed.

(* The value that the projection of a function slot gives back is
   neither null nor poison. *)
Lemma project_function_slot_not_null (statics : static_env)
    (move_from move_to : function_slot) (v w : value) :
  project_function_slot statics move_from move_to v = Some w ->
  value_at statics not_null w = true.
Proof.
  unfold project_function_slot. intros H.
  repeat
    match type of H with
    | context [match ?x with _ => _ end] => destruct x; try discriminate H
    end;
    injection H as <-; reflexivity.
Qed.

(* ---- What a defining expression makes ---------------------------------- *)

Lemma eval_simples_one (vars : env) (s : simple) (v : value) :
  eval_simples vars [s] = Some [v] -> eval_simple vars s = Some v.
Proof.
  cbn [eval_simples]. destruct (eval_simple vars s) as [w | ];
    [ | discriminate ].
  cbv beta iota. intros [= <-]. reflexivity.
Qed.

(* The value of a primitive, by cases on the primitives of which
   [stated_of_named] says something. *)
Lemma stated_of_prim_sound (statics : static_env) (h h' : heap)
    (S : stated_in_scope) (vars : env) (p : prim) (args : list simple)
    (vs : list value) (v : value) :
  stated_hold statics S vars ->
  eval_simples vars args = Some vs ->
  exec_prim statics h p vs = Some (v, h') ->
  value_at statics (stated_of_named S (Prim p args)) v = true.
Proof.
  intros Hhold Hargs Hexec.
  destruct p as [u | b | vp | hp | rp]; cbn [stated_of_named].
  - destruct u as
      [kind mut field | | | | akind | | okind | ikind op | src dst | | | n
      | | | ];
      try apply value_at_any_value.
    + (* [Block_load] *)
      destruct kind as [tag size field_kind].
      assert (Hps :
        value_at statics
          match args with
          | [Var b] => stated_of_field (stated_of_var S b) field
          | _ => any_value
          end v
        = true).
      { destruct args as [ | a [ | a' args]]; try apply value_at_any_value;
          destruct a as [b | sym | c]; try apply value_at_any_value.
        cbn [eval_simples eval_simple] in Hargs.
        destruct (lookup b vars) as [v1 | ] eqn:Hb; [ | discriminate Hargs ].
        cbv beta iota in Hargs. injection Hargs as <-.
        apply (block_load_of_shape statics h h' _ mut field v1 v _ Hexec).
        exact (stated_of_simple_sound statics S vars (Var b) v1 Hhold Hb). }
      destruct field_kind; [ exact Hps | ].
      match goal with
      | |- context [if ?test then _ else _] => destruct test
      end;
        [ exact Hps | ].
      apply tagged_value_at.
      exact (block_load_immediate statics h h' tag size mut field vs v Hexec).
    + exact (array_length_tagged statics h h' akind vs v Hexec).
    + (* [Opaque_identity] *)
      destruct args as [ | s [ | s' args]]; try apply value_at_any_value.
      rewrite (opaque_identity_same statics h h' okind vs v Hexec) in Hargs.
      apply (stated_of_simple_sound statics S vars s v Hhold).
      exact (eval_simples_one vars s v Hargs).
    + destruct ikind; try apply value_at_any_value.
      exact (unary_int_arith_tagged statics h h' op vs v Hexec).
    + destruct dst; try apply value_at_any_value.
      exact (num_conv_tagged statics h h' src vs v Hexec).
    + exact (boolean_not_tagged statics h h' vs v Hexec).
    + exact (box_number_boxed statics h h' n vs v Hexec).
    + exact (tag_immediate_tagged statics h h' vs v Hexec).
  - destruct b as [akind load_kind mut | | | ikind op | ikind op | ];
      try apply value_at_any_value.
    + destruct load_kind; try apply value_at_any_value.
      exact (array_load_immediates_tagged statics h h' akind mut vs v Hexec).
    + destruct ikind; try apply value_at_any_value.
      exact (int_arith_tagged statics h h' op vs v Hexec).
    + destruct ikind; try apply value_at_any_value.
      exact (int_shift_tagged statics h h' op vs v Hexec).
  - pose proof (exec_pure_variadic statics h h' vp vs v Hexec) as Hv.
    destruct vp as [[tag] | akind]; cbn [eval_variadic_primitive] in Hv.
    + destruct (forallb (fun w => kind_eqb (kind_of_value w) Value) vs)
        eqn:Hkinds; [ | discriminate Hv ].
      destruct (forallb (bound_if_symbol statics) vs); [ | discriminate Hv ].
      injection Hv as <-.
      unfold value_at.
      cbn [fst snd value_of_subkind block_of block_of_shapes].
      rewrite Z.eqb_refl.
      exact (fields_stated_sound statics S vars Hhold args vs Hargs Hkinds).
    + destruct (array_made akind vs) as [elements | ]; [ | discriminate Hv ].
      cbv beta iota in Hv.
      destruct (forallb (bound_if_symbol statics) vs); [ | discriminate Hv ].
      injection Hv as <-. reflexivity.
  - unfold exec_prim in Hexec.
    destruct (forallb (argument_bound statics h) vs); [ | discriminate Hexec ].
    cbn [exec_bound_prim] in Hexec.
    destruct hp as [mut tag | | akind | ]; try apply value_at_any_value;
      cbn [exec_heap_primitive] in Hexec.
    + destruct (forallb (fun w => kind_eqb (kind_of_value w) Value) vs);
        [ | discriminate Hexec ].
      injection Hexec as <- _. reflexivity.
    + destruct vs as [ | v1 vs]; [ discriminate Hexec | ].
      destruct (array_made akind (v1 :: vs)); [ | discriminate Hexec ].
      cbv beta iota in Hexec. injection Hexec as <- _. reflexivity.
  - apply value_at_any_value.
Qed.

Theorem stated_of_named_sound : stated_of_named_sound_statement.
Proof.
  intros statics hp hp' rd rd' S vars d v Hhold Hexec.
  destruct d as [s | p args | code slots | slot decls slots | | | ];
    cbn [stated_of_named]; try apply value_at_any_value.
  - cbn [exec_named eval_named] in Hexec.
    destruct (eval_simple vars s) as [w | ] eqn:Hs; [ | discriminate Hexec ].
    cbv beta iota in Hexec. injection Hexec as <- _ _.
    exact (stated_of_simple_sound statics S vars s w Hhold Hs).
  - apply FlambdaSemanticsProofs.exec_named_prim_some in Hexec
      as (vs & Hargs & Hprim & _).
    exact
      (stated_of_prim_sound statics hp hp' S vars p args vs v Hhold Hargs
         Hprim).
  - cbn [exec_named eval_named] in Hexec.
    destruct (eval_value_slots vars slots); [ | discriminate Hexec ].
    cbv beta iota in Hexec. injection Hexec as <- _ _. reflexivity.
  - cbn [exec_named eval_named] in Hexec.
    destruct (lookup slot decls); [ | discriminate Hexec ].
    cbv beta iota in Hexec.
    destruct (eval_value_slots vars slots); [ | discriminate Hexec ].
    cbv beta iota in Hexec. injection Hexec as <- _ _. reflexivity.
  - cbn [exec_named eval_named] in Hexec.
    match type of Hexec with
    | context [eval_simple vars ?c] =>
      destruct (eval_simple vars c) as [w | ]; [ | discriminate Hexec ]
    end.
    cbv beta iota in Hexec.
    match type of Hexec with
    | context [project_function_slot statics ?f ?t w] =>
      destruct (project_function_slot statics f t w) as [w' | ] eqn:Hslot;
        [ | discriminate Hexec ]
    end.
    cbv beta iota in Hexec. injection Hexec as <- _ _.
    exact (project_function_slot_not_null statics _ _ w w' Hslot).
Qed.

(* ---- The binders ------------------------------------------------------- *)

Theorem stated_hold_let : stated_hold_let_statement.
Proof.
  intros S x kind d body vars conts statics hp rd s Hhold.
  rewrite FlambdaSemanticsProofs.step_let_exec.
  destruct (exec_named statics hp rd vars d) as [[[v hp'] rd'] | ] eqn:Hexec;
    [ | discriminate ].
  destruct (kind_eqb (kind_of_value v) kind); [ | discriminate ].
  intros [= <-]. cbn [state_statics state_vars].
  apply stated_hold_cons; [ exact Hhold | ].
  exact
    (stated_of_named_sound statics hp hp' rd rd' S vars d v Hhold Hexec).
Qed.

(* Parameters bound to values of what is stated of them, around what
   was known. *)
Lemma params_stated_bound (statics : static_env) :
  ∀ (params : list variable) (vs : list value) (sks : list param_subkind)
    (S : stated_in_scope) (vars bound : env),
    stated_hold statics S vars ->
    values_of_subkinds statics vs sks = true ->
    bind_params params vs vars = Some bound ->
    stated_hold statics (params_stated params sks S) bound.
Proof.
  induction params as [ | param params IH];
    intros [ | v vs] sks S vars bound Hhold Hvs; cbn [bind_params];
    try discriminate.
  - intros [= <-]. destruct sks; exact Hhold.
  - destruct sks as [ | ps sks]; cbn [params_stated].
    + apply IH; [ | reflexivity ].
      apply stated_hold_cons; [ exact Hhold | apply value_at_any_value ].
    + rewrite FlambdaSubkindProofs.values_of_subkinds_cons in Hvs.
      apply andb_true_iff in Hvs as [Hv Hvs].
      apply IH; [ | exact Hvs ].
      apply stated_hold_cons; [ exact Hhold | exact Hv ].
Qed.

Theorem params_stated_hold : params_stated_hold_statement.
Proof.
  intros S params metadata handler def_vars def_conts recursive vs statics hp
    rd s Hhold.
  cbn [apply_cont_value].
  destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
    [ | discriminate ].
  destruct (values_of_subkinds statics vs (hm_params_subkinds metadata))
    eqn:Hvs; [ | discriminate ].
  destruct (bind_params params vs def_vars) as [handler_vars | ] eqn:Hbind;
    [ | discriminate ].
  intros [= <-]. cbn [state_statics state_vars].
  exact
    (params_stated_bound statics params vs _ S def_vars handler_vars Hhold
       Hvs Hbind).
Qed.

Theorem code_stated_hold : code_stated_hold_statement.
Proof.
  intros closure params my_closure ret exn metadata body vs args_arity
    return_arity return_to exn_to statics hp rd s Hclosure Hcall.
  pose proof
    (FlambdaSubkindBinderProofs.call_binds_at_subkinds closure _ vs
       args_arity return_arity return_to exn_to statics hp rd s Hcall)
    as Hvs.
  unfold call_code in Hcall.
  repeat
    match type of Hcall with
    | (if ?test then Undefined else _) = _ =>
      destruct test; [ discriminate Hcall | ]
    end.
  cbn [code_params code_my_closure code_body] in Hcall.
  destruct (bind_params params vs [(my_closure, closure)])
    as [callee_vars | ] eqn:Hbind; [ | discriminate Hcall ].
  injection Hcall as <-. cbn [state_statics state_vars]. unfold code_stated.
  apply
    (params_stated_bound statics params vs _ [(my_closure, not_null)]
       [(my_closure, closure)] callee_vars);
    [ | exact Hvs | exact Hbind ].
  apply stated_hold_cons; [ apply stated_hold_nil | exact Hclosure ].
Qed.

(* The same from what FlambdaContextProofs.code_refines_stated gives
   of an entry: the callee is a closure, so neither null nor poison. *)
Lemma closure_of_not_null (statics : static_env) (v : value) :
  closure_of statics v <> None -> value_at statics not_null v = true.
Proof.
  destruct v; try reflexivity; intros H; exfalso; apply H; reflexivity.
Qed.

Lemma code_stated_bound (statics : static_env) (params : list variable)
    (my_closure : variable) (metadata : code_metadata) (vs : list value)
    (callee : value) (callee_vars : env) :
  closure_of statics callee <> None ->
  values_of_subkinds statics vs (cm_params_subkinds metadata) = true ->
  bind_params params vs [(my_closure, callee)] = Some callee_vars ->
  stated_hold statics (code_stated params my_closure metadata) callee_vars.
Proof.
  intros Hclosure Hvs Hbind. unfold code_stated.
  apply
    (params_stated_bound statics params vs _ [(my_closure, not_null)]
       [(my_closure, callee)] callee_vars);
    [ | exact Hvs | exact Hbind ].
  apply stated_hold_cons;
    [ apply stated_hold_nil | apply closure_of_not_null, Hclosure ].
Qed.

(* ======================================================================== *)
(* The test of a rewrite is the test of the machine                         *)
(* ======================================================================== *)

Theorem is_stated_sound : is_stated_sound_statement.
Proof.
  intros statics S vars ps s v Hhold Hstated Heval.
  pose proof (stated_of_simple_sound statics S vars s v Hhold Heval)
    as Hknown.
  destruct s as [x | sym | c]; cbn [is_stated] in Hstated.
  - exact (value_at_weaken statics _ ps v Hstated Hknown).
  - exact (value_at_weaken statics _ ps v Hstated Hknown).
  - cbn [eval_simple] in Heval. injection Heval as <-.
    rewrite value_at_const. exact Hstated.
Qed.

Theorem are_stated_sound : are_stated_sound_statement.
Proof.
  intros statics S vars args sks. revert args.
  induction sks as [ | ps sks IH]; intros args vs Hhold Hstated Heval.
  - apply FlambdaSubkindProofs.values_of_subkinds_nil.
  - destruct args as [ | s args]; cbn [eval_simples] in Heval.
    + injection Heval as <-.
      apply FlambdaSubkindProofs.values_of_subkinds_no_values.
    + destruct (eval_simple vars s) as [v | ] eqn:Hs; [ | discriminate Heval ].
      destruct (eval_simples vars args) as [vs' | ] eqn:Hargs;
        [ | discriminate Heval ].
      cbv beta iota in Heval. injection Heval as <-.
      cbn [are_stated] in Hstated.
      apply andb_true_iff in Hstated as [Hs' Hstated].
      rewrite FlambdaSubkindProofs.values_of_subkinds_cons.
      rewrite (is_stated_sound statics S vars ps s v Hhold Hs' Hs).
      exact (IH args vs' Hhold Hstated Hargs).
Qed.

Theorem is_stated_at_sound : is_stated_at_sound_statement.
Proof.
  intros statics S vars sks i ps s v Hhold Hstated Hnth Heval.
  unfold is_stated_at in Hstated. rewrite Hnth in Hstated.
  exact (is_stated_sound statics S vars ps s v Hhold Hstated Heval).
Qed.

(* ======================================================================== *)
(* A binder that states nothing passes                                      *)
(* ======================================================================== *)

(* The tests of the rewrites ask no more than the tests that they
   took the place of: where the binder states nothing of a parameter
   (FlambdaKindModel.says_nothing), the argument passes whatever is
   known of it. *)
Lemma is_stated_of_nothing (S : stated_in_scope) (ps : param_subkind)
    (s : simple) :
  says_nothing (fst ps) (snd ps) = true -> is_stated S ps s = true.
Proof.
  destruct ps as [nullable sk]. unfold says_nothing. cbn [fst snd].
  intros Hnothing. apply andb_true_iff in Hnothing as [-> Hsk].
  destruct sk; try discriminate Hsk.
  destruct s as [x | sym | c]; cbn [is_stated];
    [ | | exact (value_at_any_value empty_static_env (Value_const c)) ];
    unfold param_says_at_least; cbn [fst snd];
    rewrite Bool.orb_true_r;
    match goal with
    | |- true && subkind_says_at_least ?a _ = true => destruct a; reflexivity
    end.
Qed.

Lemma are_stated_of_nothing (S : stated_in_scope)
    (sks : list param_subkind) :
  ∀ args : list simple,
    subkinds_say_nothing sks = true -> are_stated S args sks = true.
Proof.
  induction sks as [ | ps sks IH]; intros [ | s args]; try reflexivity.
  unfold subkinds_say_nothing. cbn [forallb are_stated]. intros Hnothing.
  apply andb_true_iff in Hnothing as [Hps Hsks].
  rewrite (is_stated_of_nothing S ps s Hps). exact (IH args Hsks).
Qed.

Lemma is_stated_at_of_nothing (S : stated_in_scope)
    (sks : list param_subkind) (i : nat) (s : simple) :
  match nth_error sks i with
  | Some ps => says_nothing (fst ps) (snd ps)
  | None => true
  end
  = true ->
  is_stated_at S sks i s = true.
Proof.
  unfold is_stated_at. destruct (nth_error sks i) as [ps | ];
    [ apply is_stated_of_nothing | reflexivity ].
Qed.

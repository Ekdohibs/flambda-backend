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


(* Proofs of the statements of FlambdaExtcallSpec.v, and examples. This
   file needs no review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaValueSpec FlambdaPrimitiveModel
  FlambdaPrimitiveSpec FlambdaPrimitiveProofs FlambdaExtcallSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* The order of strings                                                     *)
(* ======================================================================== *)

Theorem compare_bytes_refl : compare_bytes_refl_statement.
Proof.
  intros bytes. induction bytes as [ | byte rest IH]; [ reflexivity | ].
  simpl. rewrite Z.compare_refl. exact IH.
Qed.

Theorem compare_bytes_eq : compare_bytes_eq_statement.
Proof.
  intros bytes1.
  induction bytes1 as [ | byte1 rest1 IH]; intros [ | byte2 rest2]; simpl;
    try (split; [ discriminate | discriminate ]);
    try (split; reflexivity).
  destruct (Z.compare_spec (byte1 mod 256) (byte2 mod 256))
    as [Hbyte | Hbyte | Hbyte].
  - rewrite Hbyte. split.
    + intros Hrest. f_equal. apply IH. exact Hrest.
    + intros [= Hrest]. apply IH. exact Hrest.
  - split; [ discriminate | ]. intros [= Hsame _]. lia.
  - split; [ discriminate | ]. intros [= Hsame _]. lia.
Qed.

(* A list of bytes is not changed by taking the low eight bits of each. *)
Lemma map_low_bits_in_range (bytes : list Z) :
  forallb byte_in_range bytes = true ->
  map (fun byte => byte mod 256) bytes = bytes.
Proof.
  induction bytes as [ | byte rest IH]; [ reflexivity | ].
  simpl. intros Hrange. apply andb_true_iff in Hrange as [Hbyte Hrest].
  unfold byte_in_range in Hbyte.
  apply andb_true_iff in Hbyte as [Hlow Hhigh].
  apply Z.leb_le in Hlow. apply Z.ltb_lt in Hhigh.
  rewrite Z.mod_small by lia. rewrite (IH Hrest). reflexivity.
Qed.

Theorem compare_bytes_eq_in_range : compare_bytes_eq_in_range_statement.
Proof.
  intros bytes1 bytes2 Hrange1 Hrange2.
  pose proof (compare_bytes_eq bytes1 bytes2) as Heq.
  rewrite (map_low_bits_in_range bytes1 Hrange1),
    (map_low_bits_in_range bytes2 Hrange2) in Heq.
  exact Heq.
Qed.

Theorem compare_bytes_antisym : compare_bytes_antisym_statement.
Proof.
  intros bytes1.
  induction bytes1 as [ | byte1 rest1 IH]; intros [ | byte2 rest2]; simpl;
    try reflexivity.
  rewrite (Z.compare_antisym (byte1 mod 256) (byte2 mod 256)).
  destruct (byte1 mod 256 ?= byte2 mod 256); simpl;
    [ apply IH | reflexivity | reflexivity ].
Qed.

Theorem compare_bytes_trans : compare_bytes_trans_statement.
Proof.
  intros order bytes1. revert order.
  induction bytes1 as [ | byte1 rest1 IH];
    intros order [ | byte2 rest2] [ | byte3 rest3]; simpl;
    try congruence.
  destruct (Z.compare_spec (byte1 mod 256) (byte2 mod 256)),
    (Z.compare_spec (byte2 mod 256) (byte3 mod 256)),
    (Z.compare_spec (byte1 mod 256) (byte3 mod 256));
    try lia; try congruence.
  apply IH.
Qed.

Theorem compare_bytes_eq_compat : compare_bytes_eq_compat_statement.
Proof.
  intros bytes1.
  induction bytes1 as [ | byte1 rest1 IH];
    intros [ | byte2 rest2] bytes3; simpl; try discriminate.
  - intros _. reflexivity.
  - destruct (Z.compare_spec (byte1 mod 256) (byte2 mod 256))
      as [Hbyte | _ | _];
      try discriminate.
    intros Hrest. destruct bytes3 as [ | byte3 rest3]; [ reflexivity | ].
    simpl. rewrite Hbyte.
    destruct (byte2 mod 256 ?= byte3 mod 256);
      [ apply IH; exact Hrest | reflexivity | reflexivity ].
Qed.

(* ======================================================================== *)
(* The functions among themselves                                           *)
(* ======================================================================== *)

(* A string is not an integer. *)
Lemma string_is_not_an_integer (statics : static_env) (kind : standard_int)
    (v : value) (bytes : list Z) :
  string_of statics v = Some bytes -> int_of_value kind v = None.
Proof.
  destruct v; simpl; try discriminate. intros _. apply int_of_value_symbol.
Qed.

Theorem polymorphic_comparison_of_strings :
  polymorphic_comparison_of_strings_statement.
Proof.
  intros statics c v1 v2 bytes1 bytes2 Hv1 Hv2.
  cbn [eval_extcall]. unfold compare_values.
  rewrite (string_is_not_an_integer _ _ _ _ Hv1), Hv1, Hv2. reflexivity.
Qed.

Theorem specialised_for_tagged_immediates :
  specialised_for_tagged_immediates_statement.
Proof.
  intros statics c v1 v2 i1 i2 Hv1 Hv2.
  cbn [eval_extcall]. unfold compare_values. rewrite Hv1, Hv2.
  cbv beta iota.
  rewrite eval_prim_binary.
  assert (Hlt : (i1 <? i2) = match i1 ?= i2 with Lt => true | _ => false end)
    by reflexivity.
  assert (Hle : (i1 <=? i2) = match i1 ?= i2 with Gt => false | _ => true end)
    by reflexivity.
  assert (Hgt : (i2 <? i1) = match i1 ?= i2 with Gt => true | _ => false end).
  { unfold Z.ltb. rewrite (Z.compare_antisym i1 i2).
    destruct (i1 ?= i2); reflexivity. }
  assert (Hge : (i2 <=? i1) = match i1 ?= i2 with Lt => false | _ => true end).
  { unfold Z.leb. rewrite (Z.compare_antisym i1 i2).
    destruct (i1 ?= i2); reflexivity. }
  assert (Heq : (i1 =? i2) = match i1 ?= i2 with Eq => true | _ => false end)
    by apply Z.eqb_compare.
  destruct c;
    cbn [specialised_for_tagged_immediates eval_binary_primitive
         eval_int_comp eval_int_comparison reading int_width];
    unfold eval_phys_equal;
    rewrite ?Hv1, ?Hv2; cbv beta iota;
    rewrite ?Hlt, ?Hle, ?Hgt, ?Hge, ?Heq;
    unfold value_of_answer, answer_of, naked_bool;
    destruct (i1 ?= i2);
    (eexists; split; [ reflexivity | rewrite eval_prim_unary; reflexivity ]).
Qed.

(* ======================================================================== *)
(* The laws that the primitives obey                                        *)
(* ======================================================================== *)

Theorem eval_extcall_static_values : eval_extcall_static_values_statement.
Proof.
  intros statics statics' f vs Hsame.
  destruct f as [c | c];
    destruct vs as [ | v1 [ | v2 [ | v3 vs]]]; try reflexivity;
    cbn [eval_extcall]; unfold compare_values;
    rewrite (string_of_static_values statics statics' v1)
      by (intros s ->; apply Hsame; left; reflexivity);
    rewrite (string_of_static_values statics statics' v2)
      by (intros s ->; apply Hsame; right; left; reflexivity);
    reflexivity.
Qed.

Theorem eval_extcall_statics_extend : eval_extcall_statics_extend_statement.
Proof.
  intros statics statics' f vs v Hextend.
  assert (Hstring1 : ∀ (v1 : value) (bytes : list Z),
    v1 ∈ vs ->
    string_of statics v1 = Some bytes -> string_of statics' v1 = Some bytes).
  { intros v1 bytes Hin.
    apply string_of_statics_extend.
    intros s allocated ->. apply Hextend. exact Hin. }
  destruct f as [c | c];
    destruct vs as [ | v1 [ | v2 [ | v3 vs]]]; try discriminate;
    cbn [eval_extcall]; unfold compare_values.
  - destruct (string_of statics v1) as [bytes1 | ] eqn:Hv1; [ | discriminate ].
    destruct (string_of statics v2) as [bytes2 | ] eqn:Hv2; [ | discriminate ].
    rewrite (Hstring1 v1 bytes1 (or_introl eq_refl) Hv1),
      (Hstring1 v2 bytes2 (or_intror (or_introl eq_refl)) Hv2).
    exact (fun H => H).
  - destruct (int_of_value Standard_tagged_immediate v1) as [i1 | ];
      [ destruct (int_of_value Standard_tagged_immediate v2) as [i2 | ];
        [ exact (fun H => H) | ]
      | ];
      (destruct (string_of statics v1) as [bytes1 | ] eqn:Hv1;
        [ | discriminate ];
       destruct (string_of statics v2) as [bytes2 | ] eqn:Hv2;
        [ | discriminate ];
       rewrite (Hstring1 v1 bytes1 (or_introl eq_refl) Hv1),
         (Hstring1 v2 bytes2 (or_intror (or_introl eq_refl)) Hv2);
       exact (fun H => H)).
Qed.

(* A value that is a string or an integer, if a symbol, is bound. *)
Lemma string_of_symbol_bound (statics : static_env) (s : symbol)
    (bytes : list Z) :
  string_of statics (Value_symbol s) = Some bytes ->
  lookup s (static_values statics) <> None.
Proof.
  simpl. destruct (lookup s (static_values statics)); discriminate.
Qed.

Theorem eval_extcall_symbols_bound : eval_extcall_symbols_bound_statement.
Proof.
  intros statics f vs v s.
  destruct f as [c | c];
    destruct vs as [ | v1 [ | v2 [ | v3 vs]]]; try discriminate;
    cbn [eval_extcall]; unfold compare_values.
  - destruct (string_of statics v1) as [bytes1 | ] eqn:Hv1; [ | discriminate ].
    destruct (string_of statics v2) as [bytes2 | ] eqn:Hv2; [ | discriminate ].
    intros _ [-> | [-> | []]];
      eapply string_of_symbol_bound; eassumption.
  - destruct (int_of_value Standard_tagged_immediate v1) as [i1 | ] eqn:Hi1;
      [ destruct (int_of_value Standard_tagged_immediate v2) as [i2 | ]
          eqn:Hi2;
        [ intros _ [-> | [-> | []]];
          [ rewrite int_of_value_symbol in Hi1
          | rewrite int_of_value_symbol in Hi2 ];
          discriminate
        | ]
      | ];
      (destruct (string_of statics v1) as [bytes1 | ] eqn:Hv1;
        [ | discriminate ];
       destruct (string_of statics v2) as [bytes2 | ] eqn:Hv2;
        [ | discriminate ];
       intros _ [-> | [-> | []]];
       eapply string_of_symbol_bound; eassumption).
Qed.

Theorem eval_extcall_kinds : eval_extcall_kinds_statement.
Proof.
  intros statics f vs v.
  destruct f as [c | c];
    destruct vs as [ | v1 [ | v2 [ | v3 vs]]]; try discriminate;
    cbn [eval_extcall]; unfold compare_values.
  - destruct (string_of statics v1) as [bytes1 | ] eqn:Hv1; [ | discriminate ].
    destruct (string_of statics v2) as [bytes2 | ] eqn:Hv2; [ | discriminate ].
    intros [= <-]. cbn [map].
    rewrite (string_of_kind _ _ _ Hv1), (string_of_kind _ _ _ Hv2).
    split; reflexivity.
  - destruct (int_of_value Standard_tagged_immediate v1) as [i1 | ] eqn:Hi1;
      [ destruct (int_of_value Standard_tagged_immediate v2) as [i2 | ]
          eqn:Hi2;
        [ intros [= <-]; cbn [map];
          rewrite (int_of_value_kind _ _ _ Hi1),
            (int_of_value_kind _ _ _ Hi2);
          split; reflexivity
        | ]
      | ];
      (destruct (string_of statics v1) as [bytes1 | ] eqn:Hv1;
        [ | discriminate ];
       destruct (string_of statics v2) as [bytes2 | ] eqn:Hv2;
        [ | discriminate ];
       intros [= <-]; cbn [map];
       rewrite (string_of_kind _ _ _ Hv1), (string_of_kind _ _ _ Hv2);
       split; reflexivity).
Qed.

(* An answer is -1, 0 or 1 for [compare], and 0 or 1 for a relation. *)
Lemma answer_of_cases (c : extcall_comparison) (order : comparison) :
  match c with
  | Cmp_compare =>
    answer_of c order = -1 ∨ answer_of c order = 0 ∨ answer_of c order = 1
  | _ => answer_of c order = 0 ∨ answer_of c order = 1
  end.
Proof. destruct c, order; simpl; auto. Qed.

(* A function that has a value answers of some order. *)
Lemma eval_extcall_value (statics : static_env) (f : extcall)
    (vs : list value) (v : value) :
  eval_extcall statics f vs = Some v ->
  ∃ (c : extcall_comparison) (order : comparison),
    (f = Caml_string_comparison c ∨ f = Caml_comparison c)
    ∧ v = value_of_answer c order.
Proof.
  destruct f as [c | c];
    destruct vs as [ | v1 [ | v2 [ | v3 vs]]]; try discriminate;
    cbn [eval_extcall].
  - destruct (string_of statics v1) as [bytes1 | ]; [ | discriminate ].
    destruct (string_of statics v2) as [bytes2 | ]; [ | discriminate ].
    intros [= <-]. eexists. eexists. split; [ left | ]; reflexivity.
  - destruct (compare_values statics v1 v2) as [order | ]; [ | discriminate ].
    intros [= <-]. exists c, order. split; [ right | ]; reflexivity.
Qed.

Theorem eval_extcall_answer : eval_extcall_answer_statement.
Proof.
  intros statics f vs v Heval.
  destruct (eval_extcall_value statics f vs v Heval)
    as (c & order & Hf & ->).
  exists (answer_of c order). split; [ reflexivity | ].
  pose proof (answer_of_cases c order) as Hcases.
  destruct Hf as [-> | ->]; destruct c; exact Hcases.
Qed.

Theorem eval_extcall_in_range : eval_extcall_in_range_statement.
Proof.
  intros statics f vs v Heval.
  destruct (eval_extcall_value statics f vs v Heval)
    as (c & order & _ & ->).
  destruct c, order; reflexivity.
Qed.

Theorem eval_extcall_related : eval_extcall_related_statement.
Proof.
  intros R statics statics' f vs vs' v
    ((Hconst_l & Hconst_r) & (Hnumber & _) & Hobserved) Hvs.
  assert (Hstring : ∀ (v1 v1' : value) (bytes : list Z),
    R v1 v1' ->
    string_of statics v1 = Some bytes -> string_of statics' v1' = Some bytes).
  { intros v1 v1' bytes Hr.
    exact (proj1 (proj2 (proj2 (Hobserved v1 v1' Hr))) bytes). }
  assert (Hint : ∀ (v1 v1' : value) (i : Z),
    R v1 v1' ->
    int_of_value Standard_tagged_immediate v1 = Some i -> v1' = v1).
  { intros v1 v1' i Hr Hi.
    apply int_of_value_const in Hi as [-> _]. exact (Hconst_l _ v1' Hr). }
  assert (Hnot_int : ∀ v1 v1' : value,
    R v1 v1' ->
    int_of_value Standard_tagged_immediate v1 = None ->
    int_of_value Standard_tagged_immediate v1' = None).
  { intros v1 v1' Hr Hnone.
    destruct (int_of_value Standard_tagged_immediate v1') as [i | ] eqn:Hi;
      [ | reflexivity ].
    apply int_of_value_const in Hi as [-> Hrange].
    rewrite (Hconst_r v1 _ Hr) in Hnone.
    cbn [const_of_int int_of_value] in Hnone. rewrite Hrange in Hnone.
    discriminate. }
  destruct f as [c | c];
    destruct Hvs as [ | v1 v1' vs vs' Hr1 Hvs]; try discriminate;
    destruct Hvs as [ | v2 v2' vs vs' Hr2 Hvs]; try discriminate;
    destruct Hvs as [ | v3 v3' vs vs' Hr3 Hvs]; try discriminate;
    cbn [eval_extcall]; unfold compare_values.
  - destruct (string_of statics v1) as [bytes1 | ] eqn:Hv1; [ | discriminate ].
    destruct (string_of statics v2) as [bytes2 | ] eqn:Hv2; [ | discriminate ].
    rewrite (Hstring v1 v1' bytes1 Hr1 Hv1), (Hstring v2 v2' bytes2 Hr2 Hv2).
    intros [= <-]. eexists. split; [ reflexivity | apply Hnumber ].
  - destruct (int_of_value Standard_tagged_immediate v1) as [i1 | ] eqn:Hi1.
    + rewrite (Hint v1 v1' i1 Hr1 Hi1), Hi1.
      destruct (int_of_value Standard_tagged_immediate v2) as [i2 | ]
        eqn:Hi2.
      * rewrite (Hint v2 v2' i2 Hr2 Hi2), Hi2.
        intros [= <-]. eexists. split; [ reflexivity | apply Hnumber ].
      * rewrite (Hnot_int v2 v2' Hr2 Hi2).
        destruct (string_of statics v1) as [bytes1 | ] eqn:Hv1;
          [ | discriminate ].
        rewrite (string_is_not_an_integer _ _ _ _ Hv1) in Hi1.
        discriminate.
    + rewrite (Hnot_int v1 v1' Hr1 Hi1).
      destruct (string_of statics v1) as [bytes1 | ] eqn:Hv1;
        [ | discriminate ].
      destruct (string_of statics v2) as [bytes2 | ] eqn:Hv2;
        [ | discriminate ].
      rewrite (Hstring v1 v1' bytes1 Hr1 Hv1),
        (Hstring v2 v2' bytes2 Hr2 Hv2).
      intros [= <-]. eexists. split; [ reflexivity | apply Hnumber ].
Qed.

(* ======================================================================== *)
(* Examples                                                                 *)
(* ======================================================================== *)

Section Examples.

(* Three strings: "abc", "abd" and "ab". *)
Definition strings : static_env :=
  Static_env []
    [(1%positive, Static_string [97; 98; 99]);
     (2%positive, Static_string [97; 98; 100]);
     (3%positive, Static_string [97; 98])].

Definition tagged (i : Z) : value := Value_const (Tagged_immediate i).

Definition on_strings (c : extcall_comparison) (s1 s2 : positive)
    : option value :=
  eval_extcall strings (Caml_string_comparison c)
    [Value_symbol s1; Value_symbol s2].

Example string_compare_less :
  on_strings Cmp_compare 1 2 = Some (tagged (-1)).
Proof. reflexivity. Qed.

Example string_compare_greater :
  on_strings Cmp_compare 2 1 = Some (tagged 1).
Proof. reflexivity. Qed.

Example string_compare_same :
  on_strings Cmp_compare 1 1 = Some (tagged 0).
Proof. reflexivity. Qed.

(* A string is greater than its beginning. *)
Example string_compare_prefix :
  on_strings Cmp_compare 1 3 = Some (tagged 1).
Proof. reflexivity. Qed.

Example string_equal_same : on_strings Cmp_equal 1 1 = Some (tagged 1).
Proof. reflexivity. Qed.

Example string_equal_other : on_strings Cmp_equal 1 2 = Some (tagged 0).
Proof. reflexivity. Qed.

Example string_notequal_other :
  on_strings Cmp_notequal 1 2 = Some (tagged 1).
Proof. reflexivity. Qed.

Example string_lessthan : on_strings Cmp_lessthan 3 1 = Some (tagged 1).
Proof. reflexivity. Qed.

Example string_lessequal_same :
  on_strings Cmp_lessequal 1 1 = Some (tagged 1).
Proof. reflexivity. Qed.

Example string_greaterthan_same :
  on_strings Cmp_greaterthan 1 1 = Some (tagged 0).
Proof. reflexivity. Qed.

Example string_greaterequal :
  on_strings Cmp_greaterequal 3 1 = Some (tagged 0).
Proof. reflexivity. Qed.

(* The bytes are compared without a sign: 200 is greater than 100. *)
Example string_compare_high_byte :
  eval_extcall
    (Static_env []
       [(1%positive, Static_string [200]); (2%positive, Static_string [100])])
    (Caml_string_comparison Cmp_compare)
    [Value_symbol 1%positive; Value_symbol 2%positive]
  = Some (tagged 1).
Proof. reflexivity. Qed.

Example string_compare_of_immediate :
  eval_extcall strings (Caml_string_comparison Cmp_compare)
    [Value_symbol 1%positive; tagged 1]
  = None.
Proof. reflexivity. Qed.

Example string_compare_of_one_argument :
  eval_extcall strings (Caml_string_comparison Cmp_compare)
    [Value_symbol 1%positive]
  = None.
Proof. reflexivity. Qed.

Definition polymorphic (c : extcall_comparison) (v1 v2 : value)
    : option value :=
  eval_extcall strings (Caml_comparison c) [v1; v2].

Example compare_immediates :
  polymorphic Cmp_compare (tagged 3) (tagged 5) = Some (tagged (-1)).
Proof. reflexivity. Qed.

Example compare_negative_immediates :
  polymorphic Cmp_compare (tagged (-3)) (tagged (-5)) = Some (tagged 1).
Proof. reflexivity. Qed.

Example equal_immediates :
  polymorphic Cmp_equal (tagged 3) (tagged 3) = Some (tagged 1).
Proof. reflexivity. Qed.

Example greaterequal_immediates :
  polymorphic Cmp_greaterequal (tagged 3) (tagged 5) = Some (tagged 0).
Proof. reflexivity. Qed.

Example compare_strings :
  polymorphic Cmp_compare (Value_symbol 1%positive) (Value_symbol 2%positive)
  = Some (tagged (-1)).
Proof. reflexivity. Qed.

(* What the table has not got: an immediate against a string, two
   blocks, two boxed integers, a naked number. *)
Example compare_immediate_and_string :
  polymorphic Cmp_compare (tagged 3) (Value_symbol 1%positive) = None.
Proof. reflexivity. Qed.

Example compare_blocks :
  polymorphic Cmp_compare (Value_block 0 [tagged 1])
    (Value_block 0 [tagged 1])
  = None.
Proof. reflexivity. Qed.

Example compare_boxed_integers :
  polymorphic Cmp_equal (Value_boxed_number (Naked_int64 1))
    (Value_boxed_number (Naked_int64 1))
  = None.
Proof. reflexivity. Qed.

Example compare_naked_immediates :
  polymorphic Cmp_compare (Value_const (Naked_immediate 3))
    (Value_const (Naked_immediate 5))
  = None.
Proof. reflexivity. Qed.

End Examples.

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

(* Proofs of the statements of FlambdaRecoverComparisonSpec.v. Nothing
   here needs review. The proof of the refinement follows that of
   FlambdaCseProofs.v with an invariant of its own ([comparisons_hold]):
   of what is known, the pass reads the entries of two forms only, a
   three-way comparison and the tagging of a variable, and neither
   reads a cell, so the invariant says nothing of the heap or of the
   read record. The proof of the equivariance uses the lemmas of
   FlambdaCseProofs.v on the renaming of what is known. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaKindModel FlambdaValueSpec FlambdaPrimitiveModel
  FlambdaPrimitiveSpec FlambdaPrimitiveProofs FlambdaHeapSpec
  FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaContextProofs FlambdaNominalSpec
  FlambdaNominalProofs
  FlambdaApplyProofs FlambdaWellFormedSpec FlambdaWellFormedProofs
  FlambdaNormaliseSpec FlambdaNormaliseProofs FlambdaCseSpec
  FlambdaCseProofs FlambdaRecoverComparisonSpec.
Import ListNotations.

(* ======================================================================== *)
(* The table                                                                *)
(* ======================================================================== *)

(* Two integers in the range of a width that have the same reading are
   the same. *)
Lemma reading_injective (width : Z) (s : signedness) (a b : Z) :
  in_range width a = true ->
  in_range width b = true ->
  reading width s a = reading width s b ->
  a = b.
Proof.
  unfold in_range. intros Ha Hb.
  apply andb_true_iff in Ha as [Ha_low Ha_high].
  apply andb_true_iff in Hb as [Hb_low Hb_high].
  apply Z.leb_le in Ha_low, Hb_low. apply Z.ltb_lt in Ha_high, Hb_high.
  destruct s; cbn [reading]; [ intros Hsame; exact Hsame | ].
  unfold unsigned.
  assert (Hpow : (2 ^ width = 2 * 2 ^ (width - 1))%Z).
  { destruct (Z.le_gt_cases width 0) as [Hle | Hgt].
    - assert (Hzero : (2 ^ (width - 1) = 0)%Z)
        by (apply Z.pow_neg_r; lia).
      rewrite Hzero in Ha_low, Ha_high. lia.
    - replace width with (Z.succ (width - 1)) at 1 by lia.
      apply Z.pow_succ_r. lia. }
  remember (2 ^ (width - 1))%Z as half.
  rewrite Hpow.
  destruct (Z.ltb_spec a 0), (Z.ltb_spec b 0); lia.
Qed.

(* The comparison the other way round. *)
Lemma eval_int_comparison_swap (width : Z) (c : int_comparison) (x y : Z) :
  eval_int_comparison width (swap_comparison c) x y
  = eval_int_comparison width c y x.
Proof.
  destruct c; cbn [swap_comparison eval_int_comparison]; try reflexivity;
    rewrite Z.eqb_sym; reflexivity.
Qed.

(* The table is right: the test against 0 of the result of comparing
   two integers three ways is the comparison that the table gives of
   the two integers. *)
Lemma against_zero_sound (width : Z) (s : signedness)
    (c c' : int_comparison) (a b : Z) :
  in_range width a = true ->
  in_range width b = true ->
  against_zero s c = Some c' ->
  eval_int_comparison 63 c
    (eval_int_comp width (Yielding_int_like_compare_functions s) a b) 0
  = eval_int_comparison width c' a b.
Proof.
  intros Ha Hb Hc. cbn [eval_int_comp].
  pose proof (reading_injective width s a b Ha Hb) as Hinjective.
  destruct (Z.compare_spec (reading width s a) (reading width s b))
    as [Hsame | Hless | Hgreater].
  - specialize (Hinjective Hsame). subst b.
    destruct c as [ | | [ | ] | [ | ] | [ | ] | [ | ]]; cbn in Hc;
      inversion Hc; subst c'; cbn [eval_int_comparison];
      rewrite ?Z.eqb_refl, ?Z.ltb_irrefl, ?Z.leb_refl; reflexivity.
  - assert (Hdiffer : (a =? b)%Z = false)
      by (apply Z.eqb_neq; intros ->; lia).
    assert (H1 : (reading width s a <? reading width s b)%Z = true)
      by (apply Z.ltb_lt; lia).
    assert (H2 : (reading width s b <? reading width s a)%Z = false)
      by (apply Z.ltb_ge; lia).
    assert (H3 : (reading width s a <=? reading width s b)%Z = true)
      by (apply Z.leb_le; lia).
    assert (H4 : (reading width s b <=? reading width s a)%Z = false)
      by (apply Z.leb_gt; lia).
    destruct c as [ | | [ | ] | [ | ] | [ | ] | [ | ]]; cbn in Hc;
      inversion Hc; subst c'; cbn [eval_int_comparison];
      rewrite ?Hdiffer, ?H1, ?H2, ?H3, ?H4; reflexivity.
  - assert (Hdiffer : (a =? b)%Z = false)
      by (apply Z.eqb_neq; intros ->; lia).
    assert (H1 : (reading width s a <? reading width s b)%Z = false)
      by (apply Z.ltb_ge; lia).
    assert (H2 : (reading width s b <? reading width s a)%Z = true)
      by (apply Z.ltb_lt; lia).
    assert (H3 : (reading width s a <=? reading width s b)%Z = false)
      by (apply Z.leb_gt; lia).
    assert (H4 : (reading width s b <=? reading width s a)%Z = true)
      by (apply Z.leb_le; lia).
    destruct c as [ | | [ | ] | [ | ] | [ | ] | [ | ]]; cbn in Hc;
      inversion Hc; subst c'; cbn [eval_int_comparison];
      rewrite ?Hdiffer, ?H1, ?H2, ?H3, ?H4; reflexivity.
Qed.

(* ======================================================================== *)
(* What the pass knows where it runs                                        *)
(* ======================================================================== *)

(* The two forms of entry that the pass reads
   (FlambdaRecoverComparisonSpec.comparison_held and
   tagged_comparison_held): a three-way comparison, and the tagging of
   an immediate. *)
Definition read_form (p : prim) : bool :=
  match p with
  | Binary (Int_comp _ (Yielding_int_like_compare_functions _)) => true
  | Unary Tag_immediate => true
  | _ => false
  end.

(* An entry holds where a term runs: the variable is bound to the value
   of the primitive applied to the values of its arguments. *)
Definition entry_holds (statics : static_env) (vars : env)
    (entry : variable * (prim * list simple)) : Prop :=
  ∃ (vs : list value) (v : value),
    eval_simples vars (snd (snd entry)) = Some vs
    ∧ eval_prim statics (fst (snd entry)) vs = Some v
    ∧ lookup (fst entry) vars = Some v.

(* The invariant of the pass: every entry of the two forms holds. It is
   said with [eval_prim] and has no heap: a primitive of the two forms
   reads no cell ([read_form_reads_no_cell]), so the machine runs it as
   [eval_named] has it (FlambdaSemanticsProofs.exec_named_pure), and
   its value on the same values does not change with the heap. Of the
   other entries of what is known it says nothing. *)
Definition comparisons_hold (known : computed) (vars : env)
    (statics : static_env) : Prop :=
  ∀ entry : variable * (prim * list simple),
    entry ∈ known ->
    read_form (fst (snd entry)) = true ->
    entry_holds statics vars entry.

Lemma read_form_reads_no_cell (p : prim) (args : list simple) :
  read_form p = true -> named_reads_heap (Prim p args) = false.
Proof.
  intros Hform.
  destruct p as [u | b | vp | hp | rp]; try discriminate Hform.
  - destruct u; try discriminate Hform. reflexivity.
  - destruct b; try discriminate Hform. reflexivity.
Qed.

(* Binding a variable, once what mentions it is forgotten. *)
Lemma comparisons_hold_forget (known : computed) (vars : env)
    (statics : static_env) (x : variable) (v : value) :
  comparisons_hold known vars statics ->
  comparisons_hold (forget_computed x known) ((x, v) :: vars) statics.
Proof.
  intros Hhold entry Hin Hform. unfold forget_computed in Hin.
  apply filter_In in Hin. destruct Hin as [Hin Hkept]. cbv beta in Hkept.
  apply andb_true_iff in Hkept as [Hkey Hargs].
  apply negb_true_iff, Pos.eqb_neq in Hkey. apply negb_true_iff in Hargs.
  destruct (Hhold entry Hin Hform) as (vs & r & Hvs & Hprim & Hlookup).
  exists vs, r. split; [ | split; [ exact Hprim | ] ].
  - rewrite eval_simples_extend; [ exact Hvs | ].
    intros Hfree. apply var_in_simples_sound in Hfree.
    exact (eq_true_false_abs _ Hfree Hargs).
  - simpl. destruct (Pos.eqb_spec (fst entry) x) as [Heq | _];
      [ | exact Hlookup ].
    exfalso. apply Hkey. symmetry. exact Heq.
Qed.

Lemma comparisons_hold_bind_params (statics : static_env)
    (params : list variable) :
  ∀ (known : computed) (vs : list value) (vars handler_vars : env),
    comparisons_hold known vars statics ->
    bind_params params vs vars = Some handler_vars ->
    comparisons_hold (forget_computed_vars params known) handler_vars statics.
Proof.
  induction params as [ | param params IH];
    intros known [ | v vs] vars handler_vars Hhold; simpl; try discriminate.
  - intros [= <-]. exact Hhold.
  - apply IH, comparisons_hold_forget, Hhold.
Qed.

(* A static environment that binds more: a primitive has the value it
   had. *)
Lemma comparisons_hold_extend (known : computed) (vars : env)
    (statics statics' : static_env) :
  comparisons_hold known vars statics -> statics_extend statics statics' ->
  comparisons_hold known vars statics'.
Proof.
  intros Hhold [_ Hvalues] entry Hin Hform.
  destruct (Hhold entry Hin Hform) as (vs & v & Hvs & Hprim & Hlookup).
  exists vs, v. split; [ exact Hvs | split; [ | exact Hlookup ] ].
  apply (eval_prim_statics_extend statics statics' _ vs v);
    [ | exact Hprim ].
  intros s allocated _ Hs. apply Hvalues, Hs.
Qed.

(* Where an untagging has a value, the tagging of that value is what
   was untagged. *)
Lemma tagging_of_untagged (statics : static_env) (tagged v : value) :
  eval_prim statics (Unary Untag_immediate) [tagged] = Some v ->
  eval_prim statics (Unary Tag_immediate) [v] = Some tagged.
Proof.
  rewrite !eval_prim_unary. unfold eval_unary_primitive, int_of_value.
  destruct tagged; try discriminate.
  match goal with c : const |- _ => destruct c; try discriminate end.
  simpl.
  match goal with
  | |- context [in_range ?w ?i] =>
    destruct (in_range w i) eqn:Hrange; [ | discriminate ]
  end.
  intros [= <-]. simpl. rewrite Hrange. reflexivity.
Qed.

(* Under a [Let], from the run of the machine. A new entry of one of
   the two forms reads no cell, so it has the value that [eval_named]
   gives it; so has the untagging from which
   FlambdaCseSpec.tagged_again says that a variable holds a tagging. *)
Lemma comparisons_hold_under_let (known : computed) (x : variable)
    (d : named) (vars : env) (statics : static_env) (hp hp' : heap)
    (rd rd' : read_record) (v : value) :
  comparisons_hold known vars statics ->
  exec_named statics hp rd vars d = Some (v, hp', rd') ->
  comparisons_hold (computed_under_let known x d) ((x, v) :: vars) statics.
Proof.
  intros Hhold Hd. unfold computed_under_let.
  pose proof (comparisons_hold_forget known vars statics x v Hhold)
    as Hforgot.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    try exact Hforgot.
  destruct (shareable p && negb (var_in_simples x args)) eqn:Hadd;
    [ | exact Hforgot ].
  apply andb_true_iff in Hadd as [_ Hmention].
  apply negb_true_iff in Hmention.
  intros entry [<- | Hin] Hform.
  - cbn [fst snd] in Hform.
    destruct (exec_named_pure statics hp hp' rd rd' vars (Prim p args) v
      (read_form_reads_no_cell p args Hform) Hd)
      as (Heval & _ & _).
    simpl in Heval.
    destruct (eval_simples vars args) as [vs | ] eqn:Hvs; [ | discriminate ].
    simpl in Heval.
    exists vs, v. simpl. split; [ | split; [ exact Heval | ] ].
    + rewrite eval_simples_extend; [ exact Hvs | ].
      intros Hfree. apply var_in_simples_sound in Hfree. congruence.
    + rewrite Pos.eqb_refl. reflexivity.
  - apply in_app_or in Hin as [Hagain | Hin];
      [ | exact (Hforgot entry Hin Hform) ].
    destruct (tagged_again_spec x p args entry Hagain) as (t & -> & -> & ->).
    destruct (exec_named_pure statics hp hp' rd rd' vars
      (Prim (Unary Untag_immediate) [Var t]) v eq_refl Hd)
      as (Heval & _ & _).
    assert (Hxt : Pos.eqb x t = false).
    { cbn [var_in_simples var_in_simple existsb] in Hmention.
      rewrite orb_false_r in Hmention. exact Hmention. }
    simpl in Heval.
    destruct (lookup t vars) as [tagged | ] eqn:Ht; [ | discriminate ].
    simpl in Heval.
    exists [v], tagged. cbn [fst snd].
    split; [ | split ].
    + cbn [eval_simples eval_simple lookup]. rewrite Pos.eqb_refl.
      reflexivity.
    + exact (tagging_of_untagged statics tagged v Heval).
    + cbn [lookup]. rewrite Pos.eqb_sym, Hxt. exact Ht.
Qed.

(* ======================================================================== *)
(* What is known of a variable                                              *)
(* ======================================================================== *)

Lemma first_known_sound {A : Type}
    (f : variable * (prim * list simple) -> option A) (known : computed)
    (a : A) :
  first_known f known = Some a ->
  ∃ entry : variable * (prim * list simple),
    entry ∈ known ∧ f entry = Some a.
Proof.
  induction known as [ | entry known IH]; cbn [first_known];
    [ discriminate | ].
  destruct (f entry) as [a' | ] eqn:Hentry.
  - intros [= <-]. exists entry. split; [ left; reflexivity | exact Hentry ].
  - intros Hfound. destruct (IH Hfound) as (entry' & Hin & Hf).
    exists entry'. split; [ right; exact Hin | exact Hf ].
Qed.

Lemma comparison_held_sound (u : variable)
    (entry : variable * (prim * list simple)) (kind : standard_int)
    (s : signedness) (operands : list simple) :
  comparison_held u entry = Some (kind, s, operands) ->
  entry
  = (u,
     (Binary (Int_comp kind (Yielding_int_like_compare_functions s)),
      operands)).
Proof.
  destruct entry as [y [p args]].
  destruct p as [ p | [ | | | | | kind' [c | s']] | p | p | rp ];
    cbn [comparison_held]; try discriminate.
  destruct (Pos.eqb_spec u y) as [<- | _]; [ | discriminate ].
  intros [= <- <- <-]. reflexivity.
Qed.

Lemma tagged_comparison_held_sound (known : computed) (t : variable)
    (entry : variable * (prim * list simple)) (held : comparison) :
  tagged_comparison_held known t entry = Some held ->
  ∃ u : variable,
    entry = (t, (Unary Tag_immediate, [Var u]))
    ∧ first_known (comparison_held u) known = Some held.
Proof.
  destruct entry as [y [p args]].
  destruct p as [ [ ] | p | p | p | rp ]; cbn [tagged_comparison_held];
    try discriminate.
  destruct args as [ | [u | sym | c] [ | arg args]]; try discriminate.
  destruct (Pos.eqb_spec t y) as [<- | _]; [ | discriminate ].
  intros Hheld. exists u. split; [ reflexivity | exact Hheld ].
Qed.

(* A simple known to hold the tagged result of a three-way comparison
   has that value, and the arguments of the comparison are two integers
   of its kind. *)
Lemma compared_holds (known : computed) (vars : env) (statics : static_env)
    (tagged : simple) (kind : standard_int) (s : signedness)
    (operands : list simple) :
  comparisons_hold known vars statics ->
  compared known tagged = Some (kind, s, operands) ->
  ∃ (v1 v2 : value) (a b : Z),
    eval_simples vars operands = Some [v1; v2]
    ∧ int_of_value kind v1 = Some a
    ∧ int_of_value kind v2 = Some b
    ∧ eval_simple vars tagged
      = Some
          (Value_const
             (Tagged_immediate
                (eval_int_comp (int_width kind)
                   (Yielding_int_like_compare_functions s) a b)))
    ∧ in_range 63
        (eval_int_comp (int_width kind)
           (Yielding_int_like_compare_functions s) a b)
      = true.
Proof.
  intros Hhold Hcompared.
  destruct tagged as [t | sym | c]; cbn [compared] in Hcompared;
    try discriminate.
  apply first_known_sound in Hcompared as (entry & Hentry & Hheld).
  apply tagged_comparison_held_sound in Hheld as (u & -> & Hcomparison).
  apply first_known_sound in Hcomparison as (entry' & Hentry' & Hheld').
  apply comparison_held_sound in Hheld' as ->.
  destruct (Hhold _ Hentry eq_refl) as (vs & v & Hvs & Htag & Hlookup).
  destruct (Hhold _ Hentry' eq_refl)
    as (vs' & v' & Hvs' & Hcomp & Hlookup').
  cbn [fst snd] in Hvs, Htag, Hlookup, Hvs', Hcomp, Hlookup'.
  (* The three-way comparison. *)
  destruct vs' as [ | v1 [ | v2 [ | v3 vs']]]; try discriminate Hcomp.
  rewrite eval_prim_binary in Hcomp. cbn [eval_binary_primitive] in Hcomp.
  destruct (int_of_value kind v1) as [a | ] eqn:Ha; [ | discriminate ].
  destruct (int_of_value kind v2) as [b | ] eqn:Hb; [ | discriminate ].
  remember
    (eval_int_comp (int_width kind) (Yielding_int_like_compare_functions s)
       a b)
    as r eqn:Hr.
  cbv beta iota in Hcomp. injection Hcomp as <-.
  (* Its tagging. *)
  cbn [eval_simples eval_simple] in Hvs. rewrite Hlookup' in Hvs.
  cbv beta iota in Hvs. injection Hvs as <-.
  rewrite eval_prim_unary in Htag. cbn [eval_unary_primitive] in Htag.
  cbn [int_of_value int_width] in Htag.
  destruct (in_range 63 r) eqn:Hrange; [ | discriminate ].
  cbv beta iota in Htag. injection Htag as <-.
  exists v1, v2, a, b. subst r.
  split; [ exact Hvs' | ]. split; [ exact Ha | ].
  split; [ exact Hb | ]. split; [ exact Hlookup | exact Hrange ].
Qed.

(* ======================================================================== *)
(* The test, and the comparison it becomes                                  *)
(* ======================================================================== *)

Lemma is_tagged_zero_sound (s : simple) :
  is_tagged_zero s = true -> s = Const (Tagged_immediate 0).
Proof.
  destruct s as [x | sym | [ ]]; cbn [is_tagged_zero]; try discriminate.
  intros Hzero. apply Z.eqb_eq in Hzero. subst. reflexivity.
Qed.

(* What a test gives of two tagged immediates. *)
Lemma tested_eval (statics : static_env) (p : prim) (c : int_comparison)
    (x y : Z) :
  tested p = Some c ->
  in_range 63 x = true ->
  in_range 63 y = true ->
  eval_prim statics p
    [Value_const (Tagged_immediate x); Value_const (Tagged_immediate y)]
  = Some (naked_bool (eval_int_comparison 63 c x y)).
Proof.
  intros Htested Hx Hy.
  destruct p as [ p | [ | | [ | ] | | | [ ] [c' | s]] | p | p | rp ];
    cbn [tested] in Htested; try discriminate; injection Htested as <-;
    rewrite eval_prim_binary; cbn [eval_binary_primitive];
    try unfold eval_phys_equal; cbn [int_of_value int_width];
    rewrite Hx, Hy; reflexivity.
Qed.

(* The comparison that takes the place of a test has the value of the
   test. *)
Lemma recover_one_way_holds (known : computed) (vars : env)
    (statics : static_env) (tagged zero : simple) (c : int_comparison)
    (d' : named) :
  comparisons_hold known vars statics ->
  recover_one_way known tagged zero c = Some d' ->
  ∃ r : Z,
    eval_simple vars tagged = Some (Value_const (Tagged_immediate r))
    ∧ in_range 63 r = true
    ∧ zero = Const (Tagged_immediate 0)
    ∧ eval_named statics vars d'
      = Some (naked_bool (eval_int_comparison 63 c r 0)).
Proof.
  intros Hhold. unfold recover_one_way.
  destruct (is_tagged_zero zero) eqn:Hzero; [ | discriminate ].
  apply is_tagged_zero_sound in Hzero.
  destruct (compared known tagged) as [[[kind s] operands] | ] eqn:Hcompared;
    [ | discriminate ].
  destruct (against_zero s c) as [c' | ] eqn:Hagainst; [ | discriminate ].
  intros [= <-].
  destruct (compared_holds known vars statics tagged kind s operands Hhold
    Hcompared)
    as (v1 & v2 & a & b & Hoperands & Ha & Hb & Htagged & Hrange).
  exists
    (eval_int_comp (int_width kind) (Yielding_int_like_compare_functions s)
       a b).
  split; [ exact Htagged | ]. split; [ exact Hrange | ].
  split; [ exact Hzero | ].
  cbn [eval_named]. rewrite Hoperands. cbv beta iota.
  rewrite eval_prim_binary. cbn [eval_binary_primitive].
  rewrite Ha, Hb. cbv beta iota.
  destruct (int_of_value_const kind v1 a Ha) as [_ Ha_range].
  destruct (int_of_value_const kind v2 b Hb) as [_ Hb_range].
  rewrite (against_zero_sound (int_width kind) s c c' a b Ha_range Hb_range
    Hagainst).
  reflexivity.
Qed.

Lemma recover_comparison_sound (known : computed) (vars : env)
    (statics : static_env) (d : named) (v : value) :
  comparisons_hold known vars statics ->
  eval_named statics vars d = Some v ->
  eval_named statics vars (recover_comparison known d) = Some v.
Proof.
  intros Hhold Hd. unfold recover_comparison.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    try exact Hd.
  destruct args as [ | arg1 [ | arg2 [ | arg3 args]]]; try exact Hd.
  destruct (tested p) as [c | ] eqn:Htested; [ | exact Hd ].
  destruct (recover_one_way known arg1 arg2 c) as [d' | ] eqn:Hone.
  - destruct (recover_one_way_holds known vars statics arg1 arg2 c d' Hhold
      Hone)
      as (r & Htagged & Hrange & -> & Hd').
    rewrite Hd'. rewrite <- Hd.
    cbn [eval_named eval_simples eval_simple]. rewrite Htagged.
    cbv beta iota.
    symmetry. apply tested_eval; [ exact Htested | exact Hrange | ].
    reflexivity.
  - destruct (recover_one_way known arg2 arg1 (swap_comparison c))
      as [d' | ] eqn:Hother; [ | exact Hd ].
    destruct (recover_one_way_holds known vars statics arg2 arg1
      (swap_comparison c) d' Hhold Hother)
      as (r & Htagged & Hrange & -> & Hd').
    rewrite Hd'. rewrite <- Hd. rewrite eval_int_comparison_swap.
    cbn [eval_named eval_simples eval_simple]. rewrite Htagged.
    cbv beta iota.
    symmetry. apply tested_eval; [ exact Htested | | exact Hrange ].
    reflexivity.
Qed.

(* ======================================================================== *)
(* What the machine runs                                                    *)
(* ======================================================================== *)

(* A test is run on two tagged immediates and what takes its place on
   two integers: no argument is a location, so each runs as
   [eval_named] has it, and the heap and the read record are as they
   came (FlambdaSemanticsSpec.exec_named_no_location_statement).
   [Phys_equal] is among the tests, and on a heap it compares
   locations: not here. *)
Lemma tested_exec (statics : static_env) (hp : heap) (rd : read_record)
    (vars : env) (p : prim) (c : int_comparison) (arg1 arg2 : simple)
    (x y : Z) :
  tested p = Some c ->
  eval_simple vars arg1 = Some (Value_const (Tagged_immediate x)) ->
  eval_simple vars arg2 = Some (Value_const (Tagged_immediate y)) ->
  exec_named statics hp rd vars (Prim p [arg1; arg2])
  = let* v := eval_named statics vars (Prim p [arg1; arg2]) in
    Some (v, hp, rd).
Proof.
  intros Htested Harg1 Harg2.
  apply (exec_named_no_location statics hp rd vars p [arg1; arg2]
    [Value_const (Tagged_immediate x); Value_const (Tagged_immediate y)]).
  - intros q ->. discriminate Htested.
  - cbn [eval_simples]. rewrite Harg1, Harg2. reflexivity.
  - intros l [Hl | [Hl | []]]; discriminate Hl.
Qed.

Lemma recover_one_way_exec (known : computed) (vars : env)
    (statics : static_env) (hp : heap) (rd : read_record)
    (tagged zero : simple) (c : int_comparison) (d' : named) :
  comparisons_hold known vars statics ->
  recover_one_way known tagged zero c = Some d' ->
  exec_named statics hp rd vars d'
  = let* v := eval_named statics vars d' in
    Some (v, hp, rd).
Proof.
  intros Hhold. unfold recover_one_way.
  destruct (is_tagged_zero zero); [ | discriminate ].
  destruct (compared known tagged) as [[[kind s] operands] | ] eqn:Hcompared;
    [ | discriminate ].
  destruct (against_zero s c) as [c' | ]; [ | discriminate ].
  intros [= <-].
  destruct (compared_holds known vars statics tagged kind s operands Hhold
    Hcompared)
    as (v1 & v2 & a & b & Hoperands & Ha & Hb & _ & _).
  apply (exec_named_no_location statics hp rd vars _ operands [v1; v2]);
    [ intros q; discriminate | exact Hoperands | ].
  intros l [-> | [-> | []]].
  - destruct (int_of_value_const kind _ a Ha) as [Hconst _].
    discriminate Hconst.
  - destruct (int_of_value_const kind _ b Hb) as [Hconst _].
    discriminate Hconst.
Qed.

(* What the machine gives of a named expression, it gives of what the
   pass puts in its place, with the same heap and the same read
   record. *)
Lemma recover_comparison_exec (known : computed) (vars : env)
    (statics : static_env) (hp hp' : heap) (rd rd' : read_record)
    (d : named) (v : value) :
  comparisons_hold known vars statics ->
  exec_named statics hp rd vars d = Some (v, hp', rd') ->
  exec_named statics hp rd vars (recover_comparison known d)
  = Some (v, hp', rd').
Proof.
  intros Hhold Hd.
  assert (Hruns :
    recover_comparison known d = d
    ∨ (exec_named statics hp rd vars d
       = (let* v0 := eval_named statics vars d in
          Some (v0, hp, rd))
       ∧ exec_named statics hp rd vars (recover_comparison known d)
         = (let* v0 := eval_named statics vars (recover_comparison known d) in
            Some (v0, hp, rd)))).
  { unfold recover_comparison.
    destruct d as [s | p args | code slots | fslot decls slots | slot closure
      | pfrom slot closure | move_from move_to closure];
      try (left; reflexivity).
    destruct args as [ | arg1 [ | arg2 [ | arg3 args]]];
      try (left; reflexivity).
    destruct (tested p) as [c | ] eqn:Htested; [ | left; reflexivity ].
    destruct (recover_one_way known arg1 arg2 c) as [d' | ] eqn:Hone.
    - right.
      destruct (recover_one_way_holds known vars statics arg1 arg2 c d' Hhold
        Hone)
        as (r & Htagged & _ & -> & _).
      split.
      + exact (tested_exec statics hp rd vars p c arg1
          (Const (Tagged_immediate 0)) r 0 Htested Htagged eq_refl).
      + exact (recover_one_way_exec known vars statics hp rd arg1
          (Const (Tagged_immediate 0)) c d' Hhold Hone).
    - destruct (recover_one_way known arg2 arg1 (swap_comparison c))
        as [d' | ] eqn:Hother; [ | left; reflexivity ].
      right.
      destruct (recover_one_way_holds known vars statics arg2 arg1
        (swap_comparison c) d' Hhold Hother)
        as (r & Htagged & _ & -> & _).
      split.
      + exact (tested_exec statics hp rd vars p c
          (Const (Tagged_immediate 0)) arg2 0 r Htested eq_refl Htagged).
      + exact (recover_one_way_exec known vars statics hp rd arg2
          (Const (Tagged_immediate 0)) (swap_comparison c) d' Hhold
          Hother). }
  destruct Hruns as [-> | [Hsource Htarget]]; [ exact Hd | ].
  rewrite Hsource in Hd. revert Hd.
  destruct (eval_named statics vars d) as [v0 | ] eqn:Heval;
    [ | discriminate ].
  intros Hd. injection Hd as Hv Hhp Hrd. subst v0 hp' rd'.
  rewrite Htarget,
    (recover_comparison_sound known vars statics d v Hhold Heval).
  reflexivity.
Qed.

(* ======================================================================== *)
(* The pass                                                                 *)
(* ======================================================================== *)

(* As in FlambdaCseProofs.v: at a [Let], what the machine gives of the
   named expression it gives of what the pass puts in its place; then
   the parts under binders, by the induction, with what is known
   there. *)
Fixpoint recover_comparisons_knowing_refines (e : expr) :
  ∀ (known : computed) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record),
    comparisons_hold known vars statics ->
    refines (State e vars conts statics hp rd)
      (State (recover_comparisons_knowing known e) vars conts statics hp rd)
with recover_comparisons_in_handler_refines (h : continuation_handler) :
  ∀ (known : computed) (k : continuation) (body : expr) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record),
    comparisons_hold known vars statics ->
    refines (State (Let_cont_non_recursive k h body) vars conts statics hp rd)
      (State
         (Let_cont_non_recursive k (recover_comparisons_in_handler known h)
            body)
         vars conts statics hp rd).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap
         | ac | scrutinee arms | ];
      intros known vars conts statics hp rd Hhold; simpl.
    + (* Let *)
      apply let_refines_step_forward.
      * intros v hp' rd' Hd.
        exact (recover_comparison_exec known vars statics hp hp' rd rd' d v
          Hhold Hd).
      * intros v hp' rd' Hd _. apply recover_comparisons_knowing_refines.
        exact (comparisons_hold_under_let known x d vars statics hp hp' rd
          rd' v Hhold Hd).
    + (* Let_static *)
      apply let_static_refines_step.
      destruct (eval_static_bindings vars bindings statics)
        as [extended | ] eqn:Hbindings; [ | exact I ].
      apply recover_comparisons_knowing_refines.
      exact (comparisons_hold_extend known vars statics extended Hhold
        (eval_static_bindings_extend vars bindings _ _ Hbindings)).
    + (* Let_cont_non_recursive *)
      apply (refines_trans _
        (State
           (Let_cont_non_recursive k
              (recover_comparisons_in_handler known h) body)
           vars conts statics hp rd)).
      * apply recover_comparisons_in_handler_refines, Hhold.
      * apply let_cont_non_recursive_refines_step,
          recover_comparisons_knowing_refines, Hhold.
    + (* Let_cont_recursive *)
      apply let_cont_recursive_refines_step; [ reflexivity | intros _ ].
      apply recover_comparisons_knowing_refines, Hhold.
    + apply refines_refl.
    + apply refines_refl.
    + apply refines_refl.
    + apply refines_refl.
  - destruct h as [params metadata handler].
    intros known k body vars conts statics hp rd Hhold. simpl.
    apply (let_cont_handler_refines_where (statics_extend statics)
      (statics_extend_trans statics));
      [ apply statics_extend_refl | ].
    intros vs handler_vars conts' statics' hp' rd' Hextend _ Hbind.
    apply recover_comparisons_knowing_refines.
    apply (comparisons_hold_bind_params statics' params known vs vars
      handler_vars);
      [ exact (comparisons_hold_extend known vars statics statics' Hhold
          Hextend)
      | exact Hbind ].
Qed.

Theorem recover_known_comparisons_refines :
  recover_known_comparisons_refines_statement.
Proof.
  intros e vars conts statics hp rd. apply recover_comparisons_knowing_refines.
  intros entry [].
Qed.

(* ======================================================================== *)
(* The pass commutes with renaming                                          *)
(* ======================================================================== *)

(* An entry of what is known, and a comparison, renamed. *)
Definition rename_entry (r : renaming)
    (entry : variable * (prim * list simple))
    : variable * (prim * list simple) :=
  (rename_variable r (fst entry),
   (fst (snd entry), map (rename_simple r) (snd (snd entry)))).

Definition rename_comparison (r : renaming) (held : comparison)
    : comparison :=
  (fst held, map (rename_simple r) (snd held)).

Lemma first_known_renamed {A : Type} (r : renaming)
    (f f' : variable * (prim * list simple) -> option A) (g : A -> A)
    (known : computed) :
  (∀ entry : variable * (prim * list simple),
     f' (rename_entry r entry) = option_map g (f entry)) ->
  first_known f' (rename_computed r known)
  = option_map g (first_known f known).
Proof.
  intros Hf. induction known as [ | entry known IH]; [ reflexivity | ].
  cbn [rename_computed map first_known].
  change
    (rename_variable r (fst entry),
     (fst (snd entry), map (rename_simple r) (snd (snd entry))))
    with (rename_entry r entry).
  rewrite Hf. destruct (f entry); [ reflexivity | exact IH ].
Qed.

Lemma comparison_held_renamed (r : renaming) (u : variable)
    (entry : variable * (prim * list simple)) :
  comparison_held (rename_variable r u) (rename_entry r entry)
  = option_map (rename_comparison r) (comparison_held u entry).
Proof.
  destruct entry as [y [p args]].
  destruct p as [ p | [ | | | | | kind [c | s]] | p | p | rp ];
    try reflexivity.
  cbn [rename_entry fst snd comparison_held].
  rewrite rename_variable_eqb. destruct (Pos.eqb u y); reflexivity.
Qed.

Lemma tagged_comparison_held_renamed (r : renaming) (known : computed)
    (t : variable) (entry : variable * (prim * list simple)) :
  tagged_comparison_held (rename_computed r known) (rename_variable r t)
    (rename_entry r entry)
  = option_map (rename_comparison r) (tagged_comparison_held known t entry).
Proof.
  destruct entry as [y [p args]].
  destruct p as [ [ ] | p | p | p | rp ]; try reflexivity.
  destruct args as [ | [u | sym | c] [ | arg args]]; try reflexivity.
  cbn [rename_entry fst snd map rename_simple tagged_comparison_held].
  rewrite rename_variable_eqb. destruct (Pos.eqb t y); [ | reflexivity ].
  apply first_known_renamed. intros entry.
  apply comparison_held_renamed.
Qed.

Lemma compared_renamed (r : renaming) (known : computed) (tagged : simple) :
  compared (rename_computed r known) (rename_simple r tagged)
  = option_map (rename_comparison r) (compared known tagged).
Proof.
  destruct tagged as [t | sym | c]; try reflexivity.
  cbn [rename_simple compared]. apply first_known_renamed. intros entry.
  apply tagged_comparison_held_renamed.
Qed.

Lemma is_tagged_zero_renamed (r : renaming) (s : simple) :
  is_tagged_zero (rename_simple r s) = is_tagged_zero s.
Proof. destruct s; reflexivity. Qed.

Lemma recover_one_way_renamed (r : renaming) (known : computed)
    (tagged zero : simple) (c : int_comparison) :
  recover_one_way (rename_computed r known) (rename_simple r tagged)
    (rename_simple r zero) c
  = option_map (rename_named r) (recover_one_way known tagged zero c).
Proof.
  unfold recover_one_way.
  rewrite is_tagged_zero_renamed, compared_renamed.
  destruct (is_tagged_zero zero); [ | reflexivity ].
  destruct (compared known tagged) as [[[kind s] operands] | ];
    [ | reflexivity ].
  cbn [option_map rename_comparison fst snd].
  destruct (against_zero s c); reflexivity.
Qed.

Lemma recover_comparison_renamed (r : renaming) (known : computed)
    (d : named) :
  recover_comparison (rename_computed r known) (rename_named r d)
  = rename_named r (recover_comparison known d).
Proof.
  destruct d as [s | p args | | | | | ]; try reflexivity.
  destruct args as [ | arg1 [ | arg2 [ | arg3 args]]]; try reflexivity.
  cbn [rename_named map recover_comparison].
  destruct (tested p) as [c | ]; [ | reflexivity ].
  rewrite !recover_one_way_renamed.
  destruct (recover_one_way known arg1 arg2 c); [ reflexivity | ].
  destruct (recover_one_way known arg2 arg1 (swap_comparison c));
    reflexivity.
Qed.

Fixpoint recover_comparisons_knowing_equivariant_aux (r : renaming)
    (e : expr) :
  ∀ known : computed,
    recover_comparisons_knowing (rename_computed r known) (rename_expr r e)
    = rename_expr r (recover_comparisons_knowing known e)
with recover_comparisons_in_handler_equivariant (r : renaming)
    (h : continuation_handler) :
  ∀ known : computed,
    recover_comparisons_in_handler (rename_computed r known)
      (rename_handler r h)
    = rename_handler r (recover_comparisons_in_handler known h).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros known; simpl; try reflexivity.
    + rewrite recover_comparison_renamed, computed_under_let_renamed,
        recover_comparisons_knowing_equivariant_aux.
      reflexivity.
    + rewrite recover_comparisons_knowing_equivariant_aux. reflexivity.
    + rewrite recover_comparisons_in_handler_equivariant,
        recover_comparisons_knowing_equivariant_aux.
      reflexivity.
    + rewrite recover_comparisons_knowing_equivariant_aux. reflexivity.
  - destruct h as [params metadata handler]. intros known. simpl.
    rewrite forget_computed_vars_renamed,
      recover_comparisons_knowing_equivariant_aux.
    reflexivity.
Qed.

Theorem recover_comparisons_knowing_equivariant :
  recover_comparisons_knowing_equivariant_statement.
Proof.
  intros r known e. apply recover_comparisons_knowing_equivariant_aux.
Qed.

Theorem recover_known_comparisons_equivariant :
  recover_known_comparisons_equivariant_statement.
Proof.
  intros r e. exact (recover_comparisons_knowing_equivariant_aux r e []).
Qed.

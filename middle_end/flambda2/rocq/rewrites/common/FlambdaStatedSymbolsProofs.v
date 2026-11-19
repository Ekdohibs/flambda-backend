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

(* Proofs of the statements of FlambdaStatedSymbolsSpec.v, and the
   lemmas that the rewrites which read what is known of the symbols
   use. There is no statement here, and nothing here needs review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaValueSpec FlambdaSubkindSpec
  FlambdaSubkindOrderSpec FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaStatedScopeSpec FlambdaStatedSymbolsSpec.
From OxCamlFlambda2 Require FlambdaSubkindProofs FlambdaWellFormedProofs
  FlambdaStatedScopeProofs.
Import ListNotations.

(* ======================================================================== *)
(* The invariant                                                            *)
(* ======================================================================== *)

Lemma symbols_hold_nil (statics : static_env) : symbols_hold statics [].
Proof. intros sym ps HY. discriminate HY. Qed.

Theorem symbols_hold_extend : symbols_hold_extend_statement.
Proof.
  intros statics statics' Y Hextend Hhold sym ps HY.
  exact
    (FlambdaStatedScopeProofs.value_at_extend statics statics' ps
       (Value_symbol sym) Hextend (Hhold sym ps HY)).
Qed.

Lemma stated_of_symbol_sound (statics : static_env) (Y : symbols_stated)
    (sym : symbol) :
  symbols_hold statics Y ->
  value_at statics (stated_of_symbol Y sym) (Value_symbol sym) = true.
Proof.
  intros Hhold. unfold stated_of_symbol.
  destruct (lookup sym Y) as [ps | ] eqn:HY;
    [ exact (Hhold sym ps HY) | reflexivity ].
Qed.

(* ======================================================================== *)
(* What a [Let_static] binds                                                *)
(* ======================================================================== *)

Lemma static_field_stated_sound (statics : static_env) (Y : symbols_stated)
    (vars : env) (s : simple) (ps : param_subkind) (v : value) :
  symbols_hold statics Y ->
  static_field_stated Y s = Some ps ->
  eval_simple vars s = Some v ->
  kind_eqb (kind_of_value v) Value = true ∧ value_at statics ps v = true.
Proof.
  intros Hhold Hfield Heval.
  destruct s as [x | sym | c]; cbn [static_field_stated] in Hfield.
  - discriminate Hfield.
  - injection Hfield as <-. cbn [eval_simple] in Heval.
    injection Heval as <-. split; [ reflexivity | ].
    exact (stated_of_symbol_sound statics Y sym Hhold).
  - destruct c as [ | | | | | | | | | | | | | | ]; try discriminate Hfield.
    injection Hfield as <-. split.
    + cbn [eval_simple] in Heval. injection Heval as <-. reflexivity.
    + exact
        (FlambdaStatedScopeProofs.stated_of_simple_sound statics [] vars _ v
           (FlambdaStatedScopeProofs.stated_hold_nil statics vars) Heval).
Qed.

Lemma static_fields_stated_sound (statics : static_env) (Y : symbols_stated)
    (vars : env) :
  symbols_hold statics Y ->
  ∀ (fields : list simple) (sks : field_subkinds) (vs : list value),
    static_fields_stated Y fields = Some sks ->
    eval_simples vars fields = Some vs ->
    fields_of_subkinds statics sks vs = true.
Proof.
  intros Hhold. induction fields as [ | s fields IH]; intros sks vs;
    cbn [static_fields_stated eval_simples].
  - intros [= <-] [= <-]. reflexivity.
  - destruct (static_field_stated Y s) as [ps | ] eqn:Hs; [ | discriminate ].
    destruct (static_fields_stated Y fields) as [rest | ]; [ | discriminate ].
    cbv beta iota. intros [= <-].
    destruct (eval_simple vars s) as [v | ] eqn:Hv; [ | discriminate ].
    destruct (eval_simples vars fields) as [vs' | ]; [ | discriminate ].
    cbv beta iota. intros [= <-].
    destruct (static_field_stated_sound statics Y vars s ps v Hhold Hs Hv)
      as [Hkind Hat].
    rewrite FlambdaSubkindProofs.fields_of_subkinds_cons.
    rewrite <- FlambdaSubkindProofs.value_at_eq.
    rewrite Hkind. destruct ps as [nullable sk]. cbn [fst snd].
    rewrite Hat. exact (IH rest vs' eq_refl eq_refl).
Qed.

(* A symbol bound to what a static constant evaluates to is of what
   is known of the constant. *)
Lemma stated_of_static_const_sound (statics : static_env)
    (Y : symbols_stated) (vars : env) (sym : symbol) (const : static_const)
    (allocated : statically_allocated_value) :
  symbols_hold statics Y ->
  eval_static_const vars const = Some allocated ->
  lookup sym (static_values statics) = Some allocated ->
  value_at statics (stated_of_static_const Y const) (Value_symbol sym) = true.
Proof.
  intros Hhold Heval Hbound.
  destruct const as [ | | tag fields | | | ]; try reflexivity.
  cbn [stated_of_static_const].
  destruct (static_fields_stated Y fields) as [sks | ] eqn:Hfields;
    [ | reflexivity ].
  cbn [eval_static_const] in Heval.
  destruct (eval_simples vars fields) as [vs | ] eqn:Hvs;
    [ | discriminate Heval ].
  cbv beta iota in Heval. injection Heval as <-.
  unfold value_at. cbn [fst snd value_of_subkind block_of].
  rewrite Hbound. cbn [block_of_shapes]. rewrite Z.eqb_refl.
  exact (static_fields_stated_sound statics Y vars Hhold fields sks vs
           Hfields Hvs).
Qed.

(* The static environment with a group of bindings added knows the
   symbols as [statics_stated] says. *)
Lemma statics_stated_bound (vars : env) (extended : static_env) :
  ∀ (bindings : list static_binding) (statics : static_env)
    (Y : symbols_stated),
    eval_static_bindings vars bindings statics = Some extended ->
    symbols_hold extended Y ->
    symbols_hold extended (statics_stated bindings Y).
Proof.
  induction bindings as [ | binding bindings IH]; intros statics Y Heval Hhold.
  - exact Hhold.
  - destruct binding as [id c | id | sym const];
      cbn [eval_static_bindings statics_stated] in *.
    + destruct (lookup id (static_code statics)); [ discriminate Heval | ].
      exact (IH _ Y Heval Hhold).
    + exact (IH _ Y Heval Hhold).
    + destruct (lookup sym (static_values statics)); [ discriminate Heval | ].
      destruct (eval_static_const vars const) as [allocated | ] eqn:Hconst;
        [ | discriminate Heval ].
      cbv beta iota in Heval. apply (IH _ _ Heval).
      pose proof
        (FlambdaWellFormedProofs.eval_static_bindings_extend vars bindings _
           extended Heval) as [_ Hvalues].
      assert (Hbound : lookup sym (static_values extended) = Some allocated).
      { apply Hvalues. cbn [static_values lookup]. rewrite Pos.eqb_refl.
        reflexivity. }
      intros sym' ps. cbn [lookup].
      destruct (Pos.eqb sym' sym) eqn:Hsym.
      * apply Pos.eqb_eq in Hsym. subst sym'. intros [= <-].
        exact
          (stated_of_static_const_sound extended Y vars sym const allocated
             Hhold Hconst Hbound).
      * apply Hhold.
Qed.

Theorem statics_stated_hold : statics_stated_hold_statement.
Proof.
  intros Y bindings body vars conts statics hp rd s Hhold Hstep.
  cbn [step] in Hstep.
  destruct (eval_static_bindings vars bindings statics) as [extended | ]
    eqn:Heval; [ | discriminate Hstep ].
  injection Hstep as <-. cbn [state_statics].
  apply (statics_stated_bound vars extended bindings statics Y Heval).
  exact
    (symbols_hold_extend statics extended Y
       (FlambdaWellFormedProofs.eval_static_bindings_extend vars bindings
          statics extended Heval)
       Hhold).
Qed.

(* ======================================================================== *)
(* The test of a rewrite is the test of the machine                         *)
(* ======================================================================== *)

Theorem is_stated_knowing_sound : is_stated_knowing_sound_statement.
Proof.
  intros statics Y S vars ps s v Hsymbols Hhold Hstated Heval.
  destruct s as [x | sym | c]; cbn [is_stated_knowing] in Hstated.
  - exact
      (FlambdaStatedScopeProofs.is_stated_sound statics S vars ps _ v Hhold
         Hstated Heval).
  - cbn [eval_simple] in Heval. injection Heval as <-.
    exact
      (FlambdaStatedScopeProofs.value_at_weaken statics _ ps _ Hstated
         (stated_of_symbol_sound statics Y sym Hsymbols)).
  - exact
      (FlambdaStatedScopeProofs.is_stated_sound statics S vars ps _ v Hhold
         Hstated Heval).
Qed.

Theorem are_stated_knowing_sound : are_stated_knowing_sound_statement.
Proof.
  intros statics Y S vars args sks. revert args.
  induction sks as [ | ps sks IH]; intros args vs Hsymbols Hhold Hstated Heval.
  - apply FlambdaSubkindProofs.values_of_subkinds_nil.
  - destruct args as [ | s args]; cbn [eval_simples] in Heval.
    + injection Heval as <-.
      apply FlambdaSubkindProofs.values_of_subkinds_no_values.
    + destruct (eval_simple vars s) as [v | ] eqn:Hs; [ | discriminate Heval ].
      destruct (eval_simples vars args) as [vs' | ] eqn:Hargs;
        [ | discriminate Heval ].
      cbv beta iota in Heval. injection Heval as <-.
      cbn [are_stated_knowing] in Hstated.
      apply andb_true_iff in Hstated as [Hs' Hstated].
      rewrite FlambdaSubkindProofs.values_of_subkinds_cons.
      rewrite
        (is_stated_knowing_sound statics Y S vars ps s v Hsymbols Hhold Hs'
           Hs).
      exact (IH args vs' Hsymbols Hhold Hstated Hargs).
Qed.

(* ======================================================================== *)
(* With nothing known of the symbols, and of a binder that states nothing   *)
(* ======================================================================== *)

Lemma is_stated_knowing_nil (S : stated_in_scope) (ps : param_subkind)
    (s : simple) :
  is_stated_knowing [] S ps s = is_stated S ps s.
Proof. destruct s; reflexivity. Qed.

Lemma are_stated_knowing_nil (S : stated_in_scope)
    (sks : list param_subkind) :
  ∀ args : list simple,
    are_stated_knowing [] S args sks = are_stated S args sks.
Proof.
  induction sks as [ | ps sks IH]; intros [ | s args]; try reflexivity.
  cbn [are_stated_knowing are_stated].
  rewrite is_stated_knowing_nil, IH. reflexivity.
Qed.

(* The test passes wherever FlambdaStatedScopeSpec.are_stated does:
   what is known of a symbol says at least that it is no null. *)
Lemma stated_of_symbol_not_null (statics : static_env) (Y : symbols_stated)
    (sym : symbol) (ps : param_subkind) :
  param_says_at_least not_null ps = true ->
  symbols_hold statics Y ->
  value_at statics ps (Value_symbol sym) = true.
Proof.
  intros Hps _.
  exact
    (FlambdaStatedScopeProofs.value_at_weaken statics not_null ps
       (Value_symbol sym) Hps eq_refl).
Qed.

Lemma is_stated_knowing_of_nothing (Y : symbols_stated) (S : stated_in_scope)
    (ps : param_subkind) (s : simple) :
  says_nothing (fst ps) (snd ps) = true -> is_stated_knowing Y S ps s = true.
Proof.
  intros Hnothing. destruct s as [x | sym | c]; cbn [is_stated_knowing];
    try exact (FlambdaStatedScopeProofs.is_stated_of_nothing S ps _ Hnothing).
  destruct ps as [nullable sk]. unfold says_nothing in Hnothing.
  cbn [fst snd] in Hnothing.
  apply andb_true_iff in Hnothing as [-> Hsk].
  destruct sk; try discriminate Hsk.
  unfold param_says_at_least. cbn [fst snd].
  rewrite Bool.orb_true_r.
  destruct (snd (stated_of_symbol Y sym)); reflexivity.
Qed.

Lemma are_stated_knowing_of_nothing (Y : symbols_stated)
    (S : stated_in_scope) (sks : list param_subkind) :
  ∀ args : list simple,
    subkinds_say_nothing sks = true -> are_stated_knowing Y S args sks = true.
Proof.
  induction sks as [ | ps sks IH]; intros [ | s args]; try reflexivity.
  unfold subkinds_say_nothing. cbn [forallb are_stated_knowing].
  intros Hnothing. apply andb_true_iff in Hnothing as [Hps Hsks].
  rewrite (is_stated_knowing_of_nothing Y S ps s Hps). exact (IH args Hsks).
Qed.

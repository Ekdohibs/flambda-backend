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


(* Proofs of the statements of FlambdaUnboxSpec.v. Nothing here needs
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaKindModel FlambdaValueSpec FlambdaPrimitiveModel
  FlambdaPrimitiveSpec FlambdaPrimitiveProofs FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaContextProofs FlambdaNominalSpec
  FlambdaApplyProofs FlambdaWellFormedSpec FlambdaWellFormedProofs
  FlambdaNormaliseSpec FlambdaNormaliseProofs FlambdaBlockLoadSpec
  FlambdaBlockLoadProofs FlambdaUnboxSpec.
Import ListNotations.

(* ======================================================================== *)
(* What it means for what is known to hold                                  *)
(* ======================================================================== *)

(* A value is another wrapped: the box of a constant, or the tagged
   immediate of a naked one. *)
Definition wrapped_value (how : wrapping) (contents wrapped : value)
    : Prop :=
  match how with
  | Boxed _ =>
    ∃ c : const, contents = Value_const c ∧ wrapped = Value_boxed_number c
  | Tagged =>
    ∃ i : Z,
      contents = Value_const (Naked_immediate i)
      ∧ wrapped = Value_const (Tagged_immediate i)
  end.

(* A variable known to be a wrapped number is bound to the value of what
   was wrapped, wrapped. *)
Definition var_number_holds (vars : env)
    (entry : variable * (wrapping * simple)) : Prop :=
  ∃ contents wrapped : value,
    eval_simple vars (snd (snd entry)) = Some contents
    ∧ lookup (fst entry) vars = Some wrapped
    ∧ wrapped_value (fst (snd entry)) contents wrapped.

(* A symbol known to be a boxed constant is bound to the box of it. *)
Definition symbol_number_holds (statics : static_env)
    (entry : symbol * const) : Prop :=
  lookup (fst entry) (static_values statics)
  = Some (Static_boxed_number (Value_const (snd entry))).

Definition var_numbers_hold (known : known_numbers) (vars : env) : Prop :=
  ∀ entry : variable * (wrapping * simple),
    entry ∈ known_wrapped_vars known -> var_number_holds vars entry.

Definition symbol_numbers_hold (known : known_numbers)
    (statics : static_env) : Prop :=
  ∀ entry : symbol * const,
    entry ∈ known_boxed_symbols known -> symbol_number_holds statics entry.

Definition numbers_hold (known : known_numbers) (vars : env)
    (statics : static_env) : Prop :=
  var_numbers_hold known vars ∧ symbol_numbers_hold known statics.

(* ======================================================================== *)
(* What is known stays so                                                   *)
(* ======================================================================== *)

(* Binding a variable, once what mentions it is forgotten. *)
Lemma numbers_hold_forget_var (known : known_numbers) (vars : env)
    (statics : static_env) (x : variable) (v : value) :
  numbers_hold known vars statics ->
  numbers_hold (forget_wrapped_var x known) ((x, v) :: vars) statics.
Proof.
  intros [Hvars Hsymbols]. split; [ | exact Hsymbols ].
  intros entry Hin. unfold forget_wrapped_var in Hin.
  cbn [known_wrapped_vars] in Hin. apply filter_In in Hin.
  destruct Hin as [Hin Hkept]. cbv beta in Hkept.
  apply andb_true_iff in Hkept as [Hkey Hcontents].
  apply negb_true_iff, Pos.eqb_neq in Hkey.
  apply negb_true_iff in Hcontents.
  destruct (Hvars entry Hin) as (contents & wrapped & Hc & Hlookup & Hw).
  exists contents, wrapped. split; [ | split; [ | exact Hw ] ].
  - rewrite eval_simple_extend; [ exact Hc | ].
    intros Hfree. apply var_in_simple_sound in Hfree.
    exact (eq_true_false_abs _ Hfree Hcontents).
  - simpl. destruct (Pos.eqb_spec (fst entry) x) as [Heq | _];
      [ | exact Hlookup ].
    exfalso. apply Hkey. symmetry. exact Heq.
Qed.

Lemma numbers_hold_bind_params (statics : static_env)
    (params : list variable) :
  ∀ (known : known_numbers) (vs : list value) (vars handler_vars : env),
    numbers_hold known vars statics ->
    bind_params params vs vars = Some handler_vars ->
    numbers_hold (forget_wrapped_vars params known) handler_vars statics.
Proof.
  induction params as [ | param params IH];
    intros known [ | v vs] vars handler_vars Hhold; simpl; try discriminate.
  - intros [= <-]. exact Hhold.
  - apply IH, numbers_hold_forget_var, Hhold.
Qed.

(* A static environment that binds more. *)
Lemma symbol_numbers_hold_extend (known : known_numbers)
    (statics statics' : static_env) :
  symbol_numbers_hold known statics -> statics_extend statics statics' ->
  symbol_numbers_hold known statics'.
Proof.
  intros Hsymbols [_ Hvalues] entry Hin. apply Hvalues, Hsymbols, Hin.
Qed.

(* ======================================================================== *)
(* Under a [Let]                                                            *)
(* ======================================================================== *)

Lemma number_wrapped_spec (d : named) (how : wrapping) (contents : simple) :
  number_wrapped d = Some (how, contents) ->
  (∃ kind : boxable_number,
     how = Boxed kind ∧ d = Prim (Unary (Box_number kind)) [contents])
  ∨ (how = Tagged ∧ d = Prim (Unary Tag_immediate) [contents]).
Proof.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    try discriminate.
  simpl.
  destruct (unary_applied p args) as [[u arg] | ] eqn:Happlied;
    [ | discriminate ].
  destruct (unary_applied_spec _ _ _ _ Happlied) as [-> ->].
  destruct u; simpl; try discriminate.
  - intros [= <- <-]. left. eexists. split; reflexivity.
  - intros [= <- <-]. right. auto.
Qed.

(* What is tagged is a naked immediate. *)
Lemma naked_immediate_of_int (v : value) (i : Z) :
  int_of_value Standard_naked_immediate v = Some i ->
  v = Value_const (Naked_immediate i).
Proof.
  unfold int_of_value.
  destruct v; try discriminate.
  destruct c; try discriminate.
  destruct (in_range _ _); [ | discriminate ].
  intros [= ->]. reflexivity.
Qed.

(* What tagging makes of a value. *)
Lemma eval_tag_immediate (statics : static_env) (unwrapped v : value) :
  eval_prim statics (Unary Tag_immediate) [unwrapped] = Some v ->
  ∃ i : Z,
    unwrapped = Value_const (Naked_immediate i)
    ∧ v = Value_const (Tagged_immediate i).
Proof.
  rewrite eval_prim_unary. unfold eval_unary_primitive.
  destruct (int_of_value Standard_naked_immediate unwrapped) as [i | ]
    eqn:Hi; simpl; [ | discriminate ].
  intros [= <-]. exists i.
  split; [ exact (naked_immediate_of_int _ _ Hi) | reflexivity ].
Qed.

(* What boxing makes of a value. *)
Lemma eval_box_number (statics : static_env) (kind : boxable_number)
    (unwrapped v : value) :
  eval_prim statics (Unary (Box_number kind)) [unwrapped] = Some v ->
  ∃ c : const, unwrapped = Value_const c ∧ v = Value_boxed_number c.
Proof.
  rewrite eval_prim_unary. unfold eval_unary_primitive.
  destruct unwrapped; try discriminate.
  destruct (is_number_of_kind kind c); [ | discriminate ].
  intros [= <-]. exists c. auto.
Qed.

(* What a wrapping makes of the value of its argument. *)
Lemma number_wrapped_value (statics : static_env) (vars : env) (d : named)
    (how : wrapping) (contents : simple) (v : value) :
  number_wrapped d = Some (how, contents) ->
  eval_named statics vars d = Some v ->
  ∃ unwrapped : value,
    eval_simple vars contents = Some unwrapped
    ∧ wrapped_value how unwrapped v.
Proof.
  intros Hwrapped Hd.
  destruct (number_wrapped_spec d how contents Hwrapped)
    as [(kind & -> & ->) | [-> ->]];
    simpl in Hd;
    (destruct (eval_simple vars contents) as [unwrapped | ];
      [ | discriminate ]);
    simpl in Hd; exists unwrapped; (split; [ reflexivity | ]).
  - exact (eval_box_number statics kind unwrapped v Hd).
  - exact (eval_tag_immediate statics unwrapped v Hd).
Qed.

Lemma numbers_hold_under_let (known : known_numbers) (x : variable)
    (d : named) (vars : env) (statics : static_env) (v : value) :
  numbers_hold known vars statics ->
  eval_named statics vars d = Some v ->
  numbers_hold (numbers_under_let known x d) ((x, v) :: vars) statics.
Proof.
  intros Hhold Hd. unfold numbers_under_let.
  pose proof (numbers_hold_forget_var known vars statics x v Hhold)
    as Hforgot.
  destruct (number_wrapped d) as [[how contents] | ] eqn:Hwrapped;
    [ | exact Hforgot ].
  destruct (var_in_simple x contents) eqn:Hmention; [ exact Hforgot | ].
  destruct Hforgot as [Hvars Hsymbols]. split; [ | exact Hsymbols ].
  intros entry Hin. simpl in Hin.
  destruct Hin as [<- | Hin]; [ | exact (Hvars entry Hin) ].
  destruct (number_wrapped_value statics vars d how contents v Hwrapped Hd)
    as (unwrapped & Hcontents & Hvalue).
  exists unwrapped, v. simpl. split; [ | split; [ | exact Hvalue ] ].
  - rewrite eval_simple_extend; [ exact Hcontents | ].
    intros Hfree. apply var_in_simple_sound in Hfree. congruence.
  - rewrite Pos.eqb_refl. reflexivity.
Qed.

(* ======================================================================== *)
(* Under a [Let_static]                                                     *)
(* ======================================================================== *)

(* Each boxed constant of a group that is bound is bound to its box. *)
Lemma static_numbers_bound (vars : env) (bindings : list static_binding) :
  ∀ statics extended : static_env,
    eval_static_bindings vars bindings statics = Some extended ->
    ∀ entry : symbol * const,
      entry ∈ flat_map static_number_boxed bindings ->
      symbol_number_holds extended entry.
Proof.
  induction bindings as [ | binding bindings IH]; intros statics extended;
    simpl; [ intros _ entry [] | ].
  destruct binding as [id c | id | sym const].
  - destruct (lookup id (static_code statics)); [ discriminate | ].
    simpl. apply IH.
  - simpl. apply IH.
  - destruct (lookup sym (static_values statics)); [ discriminate | ].
    destruct (eval_static_const vars const) as [allocated | ] eqn:Hconst;
      simpl; [ | discriminate ].
    intros Heval entry Hin. apply in_app_or in Hin.
    destruct Hin as [Hin | Hin]; [ | exact (IH _ _ Heval entry Hin) ].
    destruct const as [code slots | fslot syms decls slots | tag fields
      | [x | sym' | c] | bytes | elements];
      simpl in Hin; try contradiction.
    destruct Hin as [<- | []]. simpl in Hconst. injection Hconst as <-.
    unfold symbol_number_holds. simpl.
    destruct (eval_static_bindings_extend vars bindings _ _ Heval)
      as [_ Hvalues].
    apply Hvalues. simpl. rewrite Pos.eqb_refl. reflexivity.
Qed.

Lemma numbers_hold_under_statics (known : known_numbers) (vars : env)
    (statics : static_env) (bindings : list static_binding)
    (extended : static_env) :
  numbers_hold known vars statics ->
  eval_static_bindings vars bindings statics = Some extended ->
  numbers_hold (numbers_under_statics known bindings) vars extended.
Proof.
  intros [Hvars Hsymbols] Heval. split; [ exact Hvars | ].
  intros entry Hin. simpl in Hin. apply in_app_or in Hin.
  destruct Hin as [Hin | Hin].
  - exact (static_numbers_bound vars bindings statics extended Heval entry
      Hin).
  - exact (symbol_numbers_hold_extend known statics extended Hsymbols
      (eval_static_bindings_extend vars bindings _ _ Heval) entry Hin).
Qed.

(* ======================================================================== *)
(* Unwrapping a known number                                                *)
(* ======================================================================== *)

Lemma wrapping_eqb_eq (how how' : wrapping) :
  wrapping_eqb how how' = true -> how = how'.
Proof.
  unfold wrapping_eqb.
  destruct (wrapping_eq_dec how how'); [ auto | discriminate ].
Qed.

(* What a known wrapped variable is. *)
Lemma unwrapped_var_sound (known : known_numbers) (vars : env)
    (statics : static_env) (how : wrapping) (x : variable)
    (contents : simple) :
  numbers_hold known vars statics ->
  unwrapped known how (Var x) = Some contents ->
  ∃ unwrapped wrapped : value,
    eval_simple vars contents = Some unwrapped
    ∧ lookup x vars = Some wrapped
    ∧ wrapped_value how unwrapped wrapped.
Proof.
  intros [Hvars _]. simpl.
  destruct (lookup x (known_wrapped_vars known)) as [[how' contents'] | ]
    eqn:Hlookup; [ | discriminate ].
  destruct (wrapping_eqb how how') eqn:Hhow; [ | discriminate ].
  apply wrapping_eqb_eq in Hhow. subst how'. intros [= <-].
  exact (Hvars _ (lookup_in _ _ _ Hlookup)).
Qed.

(* The unwrapping, where it has a value, has that of the contents. *)
Lemma unwrapping_of_known_sound (known : known_numbers) (vars : env)
    (statics : static_env) (p : prim) (args : list simple)
    (contents : simple) (v : value) :
  numbers_hold known vars statics ->
  unwrapping_of_known known p args = Some contents ->
  eval_named statics vars (Prim p args) = Some v ->
  eval_simple vars contents = Some v.
Proof.
  intros Hhold. unfold unwrapping_of_known.
  destruct (unary_applied p args) as [[u wrapped] | ] eqn:Happlied;
    [ | discriminate ].
  destruct (unary_applied_spec _ _ _ _ Happlied) as [-> ->].
  destruct u; try discriminate;
    intros Hunwrapped Hd; simpl in Hd;
    (destruct wrapped as [x | s | c0]; [ | | discriminate ]).
  - (* [Unbox_number] of a variable *)
    destruct (unwrapped_var_sound _ _ _ _ _ _ Hhold Hunwrapped)
      as (unwrapped & boxed & Hc & Hx & c & -> & ->).
    simpl in Hd. rewrite Hx in Hd. simpl in Hd.
    rewrite eval_prim_unary in Hd. simpl in Hd.
    unfold boxable_contents in Hd.
    destruct (boxable_number_of_const c); [ | discriminate ].
    simpl in Hd. destruct (is_number_of_kind kind c); [ | discriminate ].
    rewrite Hc. exact Hd.
  - (* [Unbox_number] of a symbol *)
    simpl in Hunwrapped.
    destruct (lookup s (known_boxed_symbols known)) as [c | ] eqn:Hlookup;
      [ | discriminate ].
    destruct (is_number_of_kind kind c) eqn:Hkind; [ | discriminate ].
    injection Hunwrapped as <-.
    pose proof (proj2 Hhold _ (lookup_in _ _ _ Hlookup)) as Hs.
    unfold symbol_number_holds in Hs. simpl in Hs. simpl in Hd.
    rewrite eval_prim_unary in Hd. simpl in Hd. rewrite Hs in Hd.
    unfold boxable_contents in Hd.
    destruct (boxable_number_of_const c); [ | discriminate ].
    simpl in Hd. rewrite Hkind in Hd. exact Hd.
  - (* [Untag_immediate] of a variable *)
    destruct (unwrapped_var_sound _ _ _ _ _ _ Hhold Hunwrapped)
      as (unwrapped & tagged & Hc & Hx & i & -> & ->).
    simpl in Hd. rewrite Hx in Hd. simpl in Hd.
    rewrite eval_prim_unary in Hd. simpl in Hd.
    destruct (in_range _ i); [ | discriminate ].
    rewrite Hc. exact Hd.
  - (* [Untag_immediate] of a symbol: nothing is known *)
    discriminate Hunwrapped.
Qed.

Lemma unbox_known_named_sound (known : known_numbers) (vars : env)
    (statics : static_env) (d : named) (v : value) :
  numbers_hold known vars statics ->
  eval_named statics vars d = Some v ->
  eval_named statics vars (unbox_known_named known d) = Some v.
Proof.
  intros Hhold Hd. unfold unbox_known_named.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    try exact Hd.
  destruct (unwrapping_of_known known p args) as [contents | ]
    eqn:Hunwrapping; [ | exact Hd ].
  exact (unwrapping_of_known_sound known vars statics p args contents v
    Hhold Hunwrapping Hd).
Qed.

(* ======================================================================== *)
(* With the run of the machine                                              *)
(* ======================================================================== *)

(* The machine runs a [Let] with the heap ([exec_named]); the facts
   above are about the value alone ([eval_named]). The primitives that
   the pass records (a boxing, a tagging) and those that it replaces
   (an unboxing, an untagging) read no cell, so for them the two agree
   and the heap is left as it was; of any other the pass records
   nothing and changes nothing. *)
Lemma number_wrapped_reads_no_heap (d : named) (how : wrapping)
    (contents : simple) :
  number_wrapped d = Some (how, contents) -> named_reads_heap d = false.
Proof.
  intros Hwrapped.
  destruct (number_wrapped_spec d how contents Hwrapped)
    as [(kind & _ & ->) | [_ ->]];
    reflexivity.
Qed.

Lemma unwrapping_of_known_reads_no_heap (known : known_numbers) (p : prim)
    (args : list simple) (contents : simple) :
  unwrapping_of_known known p args = Some contents ->
  prim_reads_heap p = false.
Proof.
  unfold unwrapping_of_known.
  destruct (unary_applied p args) as [[u wrapped] | ] eqn:Happlied;
    [ | discriminate ].
  destruct (unary_applied_spec _ _ _ _ Happlied) as [-> ->].
  destruct u; try discriminate; intros _; reflexivity.
Qed.

Lemma numbers_hold_under_let_exec (known : known_numbers) (x : variable)
    (d : named) (vars : env) (statics : static_env) (hp hp' : heap)
      (rd rd' : read_record)
    (v : value) :
  numbers_hold known vars statics ->
  exec_named statics hp rd vars d = Some (v, hp', rd') ->
  numbers_hold (numbers_under_let known x d) ((x, v) :: vars) statics.
Proof.
  intros Hhold Hd.
  destruct (number_wrapped d) as [[how contents] | ] eqn:Hwrapped.
  - apply numbers_hold_under_let; [ exact Hhold | ].
    exact (proj1 (exec_named_pure statics hp hp' rd rd' vars d v
      (number_wrapped_reads_no_heap d how contents Hwrapped) Hd)).
  - unfold numbers_under_let. rewrite Hwrapped.
    exact (numbers_hold_forget_var known vars statics x v Hhold).
Qed.

Lemma unbox_known_named_exec (known : known_numbers) (vars : env)
    (statics : static_env) (hp hp' : heap) (rd rd' : read_record) (d : named)
      (v : value) :
  numbers_hold known vars statics ->
  exec_named statics hp rd vars d = Some (v, hp', rd') ->
  exec_named statics hp rd vars (unbox_known_named known d)
    = Some (v, hp', rd').
Proof.
  intros Hhold Hd. unfold unbox_known_named.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    try exact Hd.
  destruct (unwrapping_of_known known p args) as [contents | ]
    eqn:Hunwrapping; [ | exact Hd ].
  destruct (exec_named_pure statics hp hp' rd rd' vars (Prim p args) v
    (unwrapping_of_known_reads_no_heap known p args contents Hunwrapping)
    Hd) as (Heval & -> & ->).
  simpl.
  rewrite (unwrapping_of_known_sound known vars statics p args contents v
    Hhold Hunwrapping Heval).
  reflexivity.
Qed.

(* ======================================================================== *)
(* The pass                                                                 *)
(* ======================================================================== *)

(* As in FlambdaBlockLoadProofs.v: at a [Let], the unboxing has the value
   of the contents or none; then the parts under binders, by the
   induction, with what is known there. *)
Fixpoint unbox_numbers_knowing_refines (e : expr) :
  ∀ (known : known_numbers) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record),
    numbers_hold known vars statics ->
    refines (State e vars conts statics hp rd)
      (State (unbox_numbers_knowing known e) vars conts statics hp rd)
with unbox_numbers_in_handler_refines (h : continuation_handler) :
  ∀ (known : known_numbers) (k : continuation) (body : expr) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record),
    numbers_hold known vars statics ->
    refines (State (Let_cont_non_recursive k h body) vars conts statics hp rd)
      (State
         (Let_cont_non_recursive k (unbox_numbers_in_handler known h) body)
         vars conts statics hp rd).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap
         | ac | scrutinee arms | ];
      intros known vars conts statics hp rd Hhold; simpl.
    + (* Let: the machine runs [d] with the heap; the pass runs the
         same or the contents that are known, and leaves the same heap. *)
      apply let_refines_step_forward.
      * intros v hp' rd' Hd.
        exact (unbox_known_named_exec known vars statics hp hp' rd rd' d v Hhold
          Hd).
      * intros v hp' rd' Hd _. apply unbox_numbers_knowing_refines.
        exact
          (numbers_hold_under_let_exec known x d vars statics hp hp' rd rd' v
          Hhold Hd).
    + (* Let_static *)
      apply let_static_refines_step.
      destruct (eval_static_bindings vars bindings statics)
        as [extended | ] eqn:Hbindings; [ | exact I ].
      apply unbox_numbers_knowing_refines.
      exact (numbers_hold_under_statics known vars statics bindings
        extended Hhold Hbindings).
    + (* Let_cont_non_recursive *)
      apply (refines_trans _
        (State
           (Let_cont_non_recursive k (unbox_numbers_in_handler known h)
              body)
           vars conts statics hp rd)).
      * apply unbox_numbers_in_handler_refines, Hhold.
      * apply let_cont_non_recursive_refines_step,
          unbox_numbers_knowing_refines, Hhold.
    + (* Let_cont_recursive *)
      apply let_cont_recursive_refines_step; [ reflexivity | intros _ ].
      apply unbox_numbers_knowing_refines, Hhold.
    + apply refines_refl.
    + apply refines_refl.
    + apply refines_refl.
    + apply refines_refl.
  - destruct h as [params metadata handler].
    intros known k body vars conts statics hp rd Hhold. simpl.
    apply (let_cont_handler_refines_where (symbol_numbers_hold known)
      (symbol_numbers_hold_extend known));
      [ exact (proj2 Hhold) | ].
    intros vs handler_vars conts' statics' hp' rd' Hsymbols _ Hbind.
    apply unbox_numbers_knowing_refines.
    apply (numbers_hold_bind_params statics' params known vs vars
      handler_vars);
      [ split; [ exact (proj1 Hhold) | exact Hsymbols ] | exact Hbind ].
Qed.

Theorem unbox_known_numbers_refines : unbox_known_numbers_refines_statement.
Proof.
  intros e vars conts statics hp rd. apply unbox_numbers_knowing_refines.
  split; intros entry [].
Qed.

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


(* Proofs of the statements of FlambdaBlockLoadSpec.v. Nothing here needs
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaKindModel FlambdaValueSpec FlambdaPrimitiveModel
  FlambdaPrimitiveSpec FlambdaPrimitiveProofs FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaContextProofs FlambdaNominalSpec
  FlambdaApplyProofs FlambdaWellFormedSpec FlambdaWellFormedProofs
  FlambdaNormaliseSpec FlambdaNormaliseProofs FlambdaBlockLoadSpec.
Import ListNotations.

(* ======================================================================== *)
(* What it means for what is known to hold                                  *)
(* ======================================================================== *)

(* What is known of a field: the field has the value of what it is known
   to hold. *)
Definition field_holds (vars : env) (field : option simple) (v : value)
    : Prop :=
  match field with
  | Some s => eval_simple vars s = Some v
  | None => True
  end.

Definition fields_hold (vars : env) (fields : list (option simple))
    (vs : list value) : Prop :=
  Forall2 (field_holds vars) fields vs.

(* A variable known to be a block is bound to a block with that tag and
   as many fields, which are as they are known to be. *)
Definition var_block_holds (vars : env) (entry : variable * known_block)
    : Prop :=
  ∃ vs : list value,
    fields_hold vars (snd (snd entry)) vs
    ∧ lookup (fst entry) vars = Some (Value_block (fst (snd entry)) vs).

(* The same of a symbol, which is bound to a static block. *)
Definition symbol_block_holds (vars : env) (statics : static_env)
    (entry : symbol * known_block) : Prop :=
  ∃ vs : list value,
    fields_hold vars (snd (snd entry)) vs
    ∧ lookup (fst entry) (static_values statics)
      = Some (Static_block (fst (snd entry)) vs).

Definition vars_hold (known : known_blocks) (vars : env) : Prop :=
  ∀ entry : variable * known_block,
    entry ∈ known_vars known -> var_block_holds vars entry.

Definition symbols_hold (known : known_blocks) (vars : env)
    (statics : static_env) : Prop :=
  ∀ entry : symbol * known_block,
    entry ∈ known_symbols known -> symbol_block_holds vars statics entry.

Definition blocks_hold (known : known_blocks) (vars : env)
    (statics : static_env) : Prop :=
  vars_hold known vars ∧ symbols_hold known vars statics.

(* ======================================================================== *)
(* Fields                                                                   *)
(* ======================================================================== *)

(* Fields that have values are as they are. *)
Lemma eval_simples_fields_hold (vars : env) (fields : list simple) :
  ∀ vs : list value,
    eval_simples vars fields = Some vs ->
    fields_hold vars (map Some fields) vs.
Proof.
  induction fields as [ | s fields IH]; intros vs; simpl.
  - intros [= <-]. apply Forall2_nil.
  - destruct (eval_simple vars s) as [v | ] eqn:Hs; [ | discriminate ].
    destruct (eval_simples vars fields) as [rest | ]; [ | discriminate ].
    simpl. intros [= <-].
    apply Forall2_cons; [ exact Hs | exact (IH rest eq_refl) ].
Qed.

(* Binding a variable, what is still known of the fields holds. *)
Lemma fields_hold_forget (vars : env) (x : variable) (v : value)
    (fields : list (option simple)) (vs : list value) :
  fields_hold vars fields vs ->
  fields_hold ((x, v) :: vars) (map (forget_field x) fields) vs.
Proof.
  intros Hfields.
  induction Hfields as [ | field v' fields vs Hfield _ IH]; simpl;
    [ apply Forall2_nil | ].
  apply Forall2_cons; [ | exact IH ].
  destruct field as [s | ]; simpl; [ | exact I ].
  destruct (var_in_simple x s) eqn:Hmention; simpl; [ exact I | ].
  rewrite eval_simple_extend; [ exact Hfield | ].
  intros Hfree. apply var_in_simple_sound in Hfree. congruence.
Qed.

Lemma fields_hold_length (vars : env) (fields : list (option simple))
    (vs : list value) :
  fields_hold vars fields vs -> length vs = length fields.
Proof. intros Hfields. symmetry. exact (Forall2_length Hfields). Qed.

(* A field that is known has the value of what it holds. *)
Lemma fields_hold_nth (vars : env) (fields : list (option simple))
    (vs : list value) :
  fields_hold vars fields vs ->
  ∀ (n : nat) (contents : simple),
    nth_error fields n = Some (Some contents) ->
    ∃ v : value, nth_error vs n = Some v ∧ eval_simple vars contents = Some v.
Proof.
  intros Hfields.
  induction Hfields as [ | field v fields vs Hfield _ IH];
    intros n contents Hnth.
  - destruct n; discriminate Hnth.
  - destruct n as [ | n]; simpl in Hnth |- *.
    + injection Hnth as ->. exists v. split; [ reflexivity | exact Hfield ].
    + exact (IH n contents Hnth).
Qed.

(* ======================================================================== *)
(* What is known stays so                                                   *)
(* ======================================================================== *)

(* Binding a variable, once what mentions it is forgotten. *)
Lemma blocks_hold_forget_var (known : known_blocks) (vars : env)
    (statics : static_env) (x : variable) (v : value) :
  blocks_hold known vars statics ->
  blocks_hold (forget_var x known) ((x, v) :: vars) statics.
Proof.
  intros [Hvars Hsymbols]. split.
  - intros entry' Hin. simpl in Hin. apply in_map_iff in Hin.
    destruct Hin as (entry & <- & Hin). apply filter_In in Hin.
    destruct Hin as [Hin Hkey]. apply negb_true_iff, Pos.eqb_neq in Hkey.
    destruct (Hvars entry Hin) as (vs & Hfields & Hlookup).
    exists vs. simpl. split; [ apply fields_hold_forget, Hfields | ].
    destruct (Pos.eqb_spec (fst entry) x) as [Heq | _]; [ | exact Hlookup ].
    exfalso. apply Hkey. symmetry. exact Heq.
  - intros entry' Hin. simpl in Hin. apply in_map_iff in Hin.
    destruct Hin as (entry & <- & Hin).
    destruct (Hsymbols entry Hin) as (vs & Hfields & Hlookup).
    exists vs. simpl. split; [ apply fields_hold_forget, Hfields | ].
    exact Hlookup.
Qed.

Lemma blocks_hold_bind_params (statics : static_env)
    (params : list variable) :
  ∀ (known : known_blocks) (vs : list value) (vars handler_vars : env),
    blocks_hold known vars statics ->
    bind_params params vs vars = Some handler_vars ->
    blocks_hold (forget_vars params known) handler_vars statics.
Proof.
  induction params as [ | param params IH];
    intros known [ | v vs] vars handler_vars Hhold; simpl; try discriminate.
  - intros [= <-]. exact Hhold.
  - apply IH, blocks_hold_forget_var, Hhold.
Qed.

(* A static environment that binds more. *)
Lemma symbols_hold_extend (known : known_blocks) (vars : env)
    (statics statics' : static_env) :
  symbols_hold known vars statics -> statics_extend statics statics' ->
  symbols_hold known vars statics'.
Proof.
  intros Hsymbols [_ Hvalues] entry Hin.
  destruct (Hsymbols entry Hin) as (vs & Hfields & Hlookup).
  exists vs. split; [ exact Hfields | apply Hvalues, Hlookup ].
Qed.

(* ======================================================================== *)
(* Under a [Let]                                                            *)
(* ======================================================================== *)

Lemma block_made_spec (d : named) (block : known_block) :
  block_made d = Some block ->
  ∃ fields : list simple,
    d = Prim (Variadic (Make_block (Block_kind_values (fst block)))) fields
    ∧ snd block = map Some fields.
Proof.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    try discriminate.
  destruct p as [u | b | v | hp | rp]; try discriminate.
  destruct v as [kind | kind]; [ | discriminate ]. destruct kind as [tag].
  simpl. intros [= <-]. exists args. auto.
Qed.

Lemma blocks_hold_under_let (known : known_blocks) (x : variable)
    (d : named) (vars : env) (statics : static_env) (v : value) :
  blocks_hold known vars statics ->
  eval_named statics vars d = Some v ->
  blocks_hold (known_under_let known x d) ((x, v) :: vars) statics.
Proof.
  intros Hhold Hd. unfold known_under_let.
  pose proof (blocks_hold_forget_var known vars statics x v Hhold)
    as Hforgot.
  destruct (block_made d) as [block | ] eqn:Hmade; [ | exact Hforgot ].
  destruct Hforgot as [Hvars Hsymbols]. split; [ | exact Hsymbols ].
  intros entry Hin. simpl in Hin.
  destruct Hin as [<- | Hin]; [ | exact (Hvars entry Hin) ].
  destruct (block_made_spec d block Hmade) as (fields & -> & Hblock).
  simpl in Hd.
  destruct (eval_simples vars fields) as [vs | ] eqn:Hfields;
    [ | discriminate ].
  simpl in Hd. rewrite eval_prim_variadic in Hd. simpl in Hd.
  match type of Hd with
  | (if ?checks then _ else _) = _ =>
    destruct checks; [ | discriminate ]
  end.
  injection Hd as <-.
  exists vs. simpl. split.
  - rewrite Hblock.
    apply fields_hold_forget, eval_simples_fields_hold, Hfields.
  - rewrite Pos.eqb_refl. reflexivity.
Qed.

(* ======================================================================== *)
(* Under a [Let_static]                                                     *)
(* ======================================================================== *)

(* Each block of a group that is bound is bound to its fields' values. *)
Lemma static_blocks_bound (vars : env) (bindings : list static_binding) :
  ∀ statics extended : static_env,
    eval_static_bindings vars bindings statics = Some extended ->
    ∀ entry : symbol * known_block,
      entry ∈ flat_map static_block_made bindings ->
      symbol_block_holds vars extended entry.
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
    destruct const as [code slots | fslot syms decls slots | tag fields | s
      | bytes | elements];
      simpl in Hin; try contradiction.
    destruct Hin as [<- | []]. simpl in Hconst.
    destruct (eval_simples vars fields) as [vs | ] eqn:Hfields;
      [ | discriminate ].
    simpl in Hconst. injection Hconst as <-.
    exists vs. split; [ apply eval_simples_fields_hold, Hfields | ]. simpl.
    destruct (eval_static_bindings_extend vars bindings _ _ Heval)
      as [_ Hvalues].
    apply Hvalues. simpl. rewrite Pos.eqb_refl. reflexivity.
Qed.

Lemma blocks_hold_under_statics (known : known_blocks) (vars : env)
    (statics : static_env) (bindings : list static_binding)
    (extended : static_env) :
  blocks_hold known vars statics ->
  eval_static_bindings vars bindings statics = Some extended ->
  blocks_hold (known_under_statics known bindings) vars extended.
Proof.
  intros [Hvars Hsymbols] Heval. split; [ exact Hvars | ].
  intros entry Hin. simpl in Hin. apply in_app_or in Hin.
  destruct Hin as [Hin | Hin].
  - exact (static_blocks_bound vars bindings statics extended Heval entry
      Hin).
  - exact (symbols_hold_extend known vars statics extended Hsymbols
      (eval_static_bindings_extend vars bindings _ _ Heval) entry Hin).
Qed.

(* ======================================================================== *)
(* Loading from a known block                                               *)
(* ======================================================================== *)

(* What a known block is: the variable or symbol has a value that is a
   block with that tag, whose fields are as they are known to be. *)
Lemma known_block_of_sound (known : known_blocks) (vars : env)
    (statics : static_env) (s : simple) (block : known_block) :
  blocks_hold known vars statics ->
  known_block_of known s = Some block ->
  ∃ (vs : list value) (v : value),
    fields_hold vars (snd block) vs
    ∧ eval_simple vars s = Some v
    ∧ block_of statics v = Some (fst block, vs)
    ∧ is_tagged_immediate v = false.
Proof.
  intros [Hvars Hsymbols].
  destruct s as [x | sym | c]; simpl; try discriminate; intros Hlookup.
  - destruct (Hvars (x, block) (lookup_in _ _ _ Hlookup))
      as (vs & Hfields & Hx).
    simpl in Hfields, Hx.
    exists vs, (Value_block (fst block) vs). auto.
  - destruct (Hsymbols (sym, block) (lookup_in _ _ _ Hlookup))
      as (vs & Hfields & Hsym).
    simpl in Hfields, Hsym.
    exists vs, (Value_symbol sym).
    split; [ exact Hfields | split; [ reflexivity | ] ].
    split; [ | reflexivity ].
    simpl. rewrite Hsym. reflexivity.
Qed.

Lemma field_loaded_spec (kind : block_access_kind) (field : Z)
    (block : known_block) (contents : simple) :
  field_loaded kind field block = Some contents ->
  ∃ (tag size : option Z),
    kind = Block_access_values tag size Any_value
    ∧ known_is tag (fst block) = true
    ∧ known_is size (Z.of_nat (length (snd block))) = true
    ∧ (0 <=? field)%Z = true
    ∧ nth_error (snd block) (Z.to_nat field) = Some (Some contents).
Proof.
  destruct kind as [tag size [ | ]]; simpl; [ | discriminate ].
  destruct (known_is tag (fst block)
            && known_is size (Z.of_nat (length (snd block)))
            && (0 <=? field)%Z) eqn:Hchecks; [ | discriminate ].
  apply andb_true_iff in Hchecks as [Hchecks Hfield].
  apply andb_true_iff in Hchecks as [Htag Hsize].
  destruct (nth_error (snd block) (Z.to_nat field)) as [[known | ] | ];
    try discriminate.
  intros [= ->]. exists tag, size. auto 6.
Qed.

Lemma unary_applied_spec (p : prim) (args : list simple)
    (u : unary_primitive) (arg : simple) :
  unary_applied p args = Some (u, arg) -> p = Unary u ∧ args = [arg].
Proof.
  unfold unary_applied.
  destruct p as [u' | b | v | hp | rp]; try discriminate.
  destruct args as [ | arg' [ | s args]]; try discriminate.
  intros [= -> ->]. auto.
Qed.

(* The primitive, where it has a value, has the one the pass gives. *)
Lemma load_known_named_sound (known : known_blocks) (vars : env)
    (statics : static_env) (d : named) (v : value) :
  blocks_hold known vars statics ->
  eval_named statics vars d = Some v ->
  eval_named statics vars (load_known_named known d) = Some v.
Proof.
  intros Hhold Hd. unfold load_known_named.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    try exact Hd.
  destruct (unary_applied p args) as [[u block] | ] eqn:Happlied;
    [ | exact Hd ].
  destruct (known_block_of known block) as [known_block | ] eqn:Hknown;
    [ | exact Hd ].
  destruct (of_known_block u known_block) as [result | ] eqn:Hresult;
    [ | exact Hd ].
  destruct (unary_applied_spec _ _ _ _ Happlied) as [-> ->].
  destruct (known_block_of_sound _ _ _ _ _ Hhold Hknown)
    as (vs & block_value & Hfields & Hblock & Hblock_of & Himmediate).
  simpl in Hd |- *. rewrite Hblock in Hd. simpl in Hd.
  rewrite eval_prim_unary in Hd.
  destruct u; simpl in Hresult; try discriminate Hresult.
  - (* [Block_load]: not marked [Mutable] *)
    assert (Hfield : field_loaded kind field known_block = Some result)
      by (destruct mut; [ discriminate Hresult | exact Hresult .. ]).
    destruct (field_loaded_spec _ _ _ _ Hfield)
      as (tag & size & -> & Htag & Hsize & Hpos & Hnth).
    destruct (fields_hold_nth _ _ _ Hfields _ _ Hnth)
      as (loaded & Hloaded & Hcontents).
    simpl in Hd. rewrite Hblock_of in Hd. simpl in Hd.
    rewrite Htag, (fields_hold_length _ _ _ Hfields), Hsize, Hpos in Hd.
    simpl in Hd. rewrite Hloaded in Hd. simpl in Hd.
    destruct (kind_eqb (kind_of_value loaded) Value); [ | discriminate ].
    rewrite Hcontents. exact Hd.
  - (* [Is_int] *)
    injection Hresult as <-. simpl in Hd |- *.
    unfold eval_is_int in Hd. rewrite Himmediate in Hd.
    destruct variant_only.
    + rewrite Hblock_of in Hd. exact Hd.
    + unfold is_allocated in Hd. rewrite Hblock_of in Hd. simpl in Hd.
      rewrite orb_true_r in Hd. exact Hd.
  - (* [Get_tag] *)
    injection Hresult as <-. simpl in Hd |- *.
    rewrite Hblock_of in Hd. exact Hd.
Qed.

(* ======================================================================== *)
(* The pass                                                                 *)
(* ======================================================================== *)

(* At a [Let], the load has the value of the field or none; then the
   parts under binders, by the induction, with what is known there. *)
Fixpoint load_blocks_knowing_refines (e : expr) :
  ∀ (known : known_blocks) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap),
    blocks_hold known vars statics ->
    refines (State e vars conts statics hp)
      (State (load_blocks_knowing known e) vars conts statics hp)
with load_blocks_in_handler_refines (h : continuation_handler) :
  ∀ (known : known_blocks) (k : continuation) (body : expr) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap),
    blocks_hold known vars statics ->
    refines (State (Let_cont_non_recursive k h body) vars conts statics hp)
      (State
         (Let_cont_non_recursive k (load_blocks_in_handler known h) body)
         vars conts statics hp).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap
         | ac | scrutinee arms | ];
      intros known vars conts statics hp Hhold; simpl.
    + (* Let *)
      destruct (eval_named statics vars d) as [v | ] eqn:Hd;
        [ destruct (kind_eqb (kind_of_value v) kind) eqn:Hkind | ].
      * eapply refines_step;
          [ rewrite step_let, Hd, Hkind; reflexivity
          | rewrite step_let,
              (load_known_named_sound known vars statics d v Hhold Hd), Hkind;
            reflexivity
          | ].
        apply load_blocks_knowing_refines.
        exact (blocks_hold_under_let known x d vars statics v Hhold Hd).
      * apply step_undefined_refines. rewrite step_let, Hd, Hkind.
        reflexivity.
      * apply step_undefined_refines. rewrite step_let, Hd. reflexivity.
    + (* Let_static *)
      apply let_static_refines_step.
      destruct (eval_static_bindings vars bindings statics)
        as [extended | ] eqn:Hbindings; [ | exact I ].
      apply load_blocks_knowing_refines.
      exact (blocks_hold_under_statics known vars statics bindings extended
        Hhold Hbindings).
    + (* Let_cont_non_recursive *)
      apply (refines_trans _
        (State
           (Let_cont_non_recursive k (load_blocks_in_handler known h) body)
           vars conts statics hp)).
      * apply load_blocks_in_handler_refines, Hhold.
      * apply let_cont_non_recursive_refines_step,
          load_blocks_knowing_refines, Hhold.
    + (* Let_cont_recursive *)
      apply let_cont_recursive_refines_step; [ reflexivity | intros _ ].
      apply load_blocks_knowing_refines, Hhold.
    + apply refines_refl.
    + apply refines_refl.
    + apply refines_refl.
    + apply refines_refl.
  - destruct h as [params metadata handler].
    intros known k body vars conts statics hp Hhold. simpl.
    apply (let_cont_handler_refines_where (symbols_hold known vars)
      (symbols_hold_extend known vars));
      [ exact (proj2 Hhold) | ].
    intros vs handler_vars conts' statics' hp' Hsymbols _ Hbind.
    apply load_blocks_knowing_refines.
    apply (blocks_hold_bind_params statics' params known vs vars
      handler_vars);
      [ split; [ exact (proj1 Hhold) | exact Hsymbols ] | exact Hbind ].
Qed.

Theorem load_known_blocks_refines : load_known_blocks_refines_statement.
Proof.
  intros e vars conts statics hp. apply load_blocks_knowing_refines.
  split; intros entry [].
Qed.

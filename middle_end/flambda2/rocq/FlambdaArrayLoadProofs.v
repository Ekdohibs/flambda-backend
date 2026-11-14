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


(* Proofs of the statements of FlambdaArrayLoadSpec.v. Nothing here needs
   review. The refinement follows the proof for the fields of a known
   block (FlambdaBlockLoadProofs.v), whose lemmas on what is known of
   the fields of a block are used for the elements of an array. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaKindModel FlambdaValueSpec FlambdaPrimitiveModel
  FlambdaPrimitiveSpec FlambdaPrimitiveProofs FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaContextProofs FlambdaNominalSpec
  FlambdaNominalProofs FlambdaApplyProofs FlambdaWellFormedSpec
  FlambdaWellFormedProofs FlambdaNormaliseSpec FlambdaNormaliseProofs
  FlambdaValidatorEquivarianceBaseProofs FlambdaBlockLoadSpec
  FlambdaBlockLoadProofs FlambdaArrayLoadSpec.
Import ListNotations.

(* ======================================================================== *)
(* What it means for what is known to hold                                  *)
(* ======================================================================== *)

(* A variable known to be an array is bound to an array value of as
   many elements, which are as they are known to be. *)
Definition var_array_holds (vars : env) (entry : variable * known_array)
    : Prop :=
  ∃ vs : list value,
    fields_hold vars (snd entry) vs
    ∧ lookup (fst entry) vars = Some (Value_immutable_array vs).

(* The same of a symbol, which is bound to a static array. *)
Definition symbol_array_holds (vars : env) (statics : static_env)
    (entry : symbol * known_array) : Prop :=
  ∃ vs : list value,
    fields_hold vars (snd entry) vs
    ∧ lookup (fst entry) (static_values statics)
      = Some (Static_immutable_array vs).

Definition array_vars_hold (known : known_arrays) (vars : env) : Prop :=
  ∀ entry : variable * known_array,
    entry ∈ known_array_vars known -> var_array_holds vars entry.

Definition array_symbols_hold (known : known_arrays) (vars : env)
    (statics : static_env) : Prop :=
  ∀ entry : symbol * known_array,
    entry ∈ known_array_symbols known ->
    symbol_array_holds vars statics entry.

Definition arrays_hold (known : known_arrays) (vars : env)
    (statics : static_env) : Prop :=
  array_vars_hold known vars ∧ array_symbols_hold known vars statics.

(* ======================================================================== *)
(* What is known stays so                                                   *)
(* ======================================================================== *)

(* Binding a variable, once what mentions it is forgotten. *)
Lemma arrays_hold_forget_var (known : known_arrays) (vars : env)
    (statics : static_env) (x : variable) (v : value) :
  arrays_hold known vars statics ->
  arrays_hold (forget_array_var x known) ((x, v) :: vars) statics.
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

Lemma arrays_hold_bind_params (statics : static_env)
    (params : list variable) :
  ∀ (known : known_arrays) (vs : list value) (vars handler_vars : env),
    arrays_hold known vars statics ->
    bind_params params vs vars = Some handler_vars ->
    arrays_hold (forget_array_vars params known) handler_vars statics.
Proof.
  induction params as [ | param params IH];
    intros known [ | v vs] vars handler_vars Hhold; simpl; try discriminate.
  - intros [= <-]. exact Hhold.
  - apply IH, arrays_hold_forget_var, Hhold.
Qed.

(* A static environment that binds more. *)
Lemma array_symbols_hold_extend (known : known_arrays) (vars : env)
    (statics statics' : static_env) :
  array_symbols_hold known vars statics -> statics_extend statics statics' ->
  array_symbols_hold known vars statics'.
Proof.
  intros Hsymbols [_ Hvalues] entry Hin.
  destruct (Hsymbols entry Hin) as (vs & Hfields & Hlookup).
  exists vs. split; [ exact Hfields | apply Hvalues, Hlookup ].
Qed.

(* ======================================================================== *)
(* Under a [Let]                                                            *)
(* ======================================================================== *)

Lemma array_made_by_spec (d : named) (array : known_array) :
  array_made_by d = Some array ->
  ∃ (kind : array_kind) (elements : list simple),
    d = Prim (Variadic (Make_immutable_array kind)) elements
    ∧ array = map Some elements.
Proof.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    try discriminate.
  destruct p as [u | b | v | hp | rp]; try discriminate.
  destruct v as [kind | kind]; [ discriminate | ].
  simpl. intros [= <-]. exists kind, args. auto.
Qed.

Lemma arrays_hold_under_let (known : known_arrays) (x : variable)
    (d : named) (vars : env) (statics : static_env) (v : value) :
  arrays_hold known vars statics ->
  eval_named statics vars d = Some v ->
  arrays_hold (arrays_under_let known x d) ((x, v) :: vars) statics.
Proof.
  intros Hhold Hd. unfold arrays_under_let.
  pose proof (arrays_hold_forget_var known vars statics x v Hhold)
    as Hforgot.
  destruct (array_made_by d) as [array | ] eqn:Hmade; [ | exact Hforgot ].
  destruct Hforgot as [Hvars Hsymbols]. split; [ | exact Hsymbols ].
  intros entry Hin. simpl in Hin.
  destruct Hin as [<- | Hin]; [ | exact (Hvars entry Hin) ].
  destruct (array_made_by_spec d array Hmade) as (kind & elements & -> & ->).
  simpl in Hd.
  destruct (eval_simples vars elements) as [vs | ] eqn:Helements;
    [ | discriminate ].
  simpl in Hd. rewrite eval_prim_variadic in Hd. simpl in Hd.
  unfold array_made in Hd.
  destruct (forallb (array_element_ok kind) vs); [ | discriminate ].
  simpl in Hd.
  destruct (forallb (bound_if_symbol statics) vs); [ | discriminate ].
  injection Hd as <-.
  exists vs. simpl. split.
  - apply fields_hold_forget, eval_simples_fields_hold, Helements.
  - rewrite Pos.eqb_refl. reflexivity.
Qed.

(* ======================================================================== *)
(* Under a [Let_static]                                                     *)
(* ======================================================================== *)

(* Each array of a group that is bound is bound to its elements'
   values. *)
Lemma static_arrays_bound (vars : env) (bindings : list static_binding) :
  ∀ statics extended : static_env,
    eval_static_bindings vars bindings statics = Some extended ->
    ∀ entry : symbol * known_array,
      entry ∈ flat_map static_array_made bindings ->
      symbol_array_holds vars extended entry.
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
    destruct (eval_simples vars elements) as [vs | ] eqn:Helements;
      [ | discriminate ].
    simpl in Hconst. injection Hconst as <-.
    exists vs. split; [ apply eval_simples_fields_hold, Helements | ]. simpl.
    destruct (eval_static_bindings_extend vars bindings _ _ Heval)
      as [_ Hvalues].
    apply Hvalues. simpl. rewrite Pos.eqb_refl. reflexivity.
Qed.

Lemma arrays_hold_under_statics (known : known_arrays) (vars : env)
    (statics : static_env) (bindings : list static_binding)
    (extended : static_env) :
  arrays_hold known vars statics ->
  eval_static_bindings vars bindings statics = Some extended ->
  arrays_hold (arrays_under_statics known bindings) vars extended.
Proof.
  intros [Hvars Hsymbols] Heval. split; [ exact Hvars | ].
  intros entry Hin. simpl in Hin. apply in_app_or in Hin.
  destruct Hin as [Hin | Hin].
  - exact (static_arrays_bound vars bindings statics extended Heval entry
      Hin).
  - exact (array_symbols_hold_extend known vars statics extended Hsymbols
      (eval_static_bindings_extend vars bindings _ _ Heval) entry Hin).
Qed.

(* ======================================================================== *)
(* Reading a known array                                                    *)
(* ======================================================================== *)

(* What a known array is: the variable or symbol has a value that is
   read as an array, whose elements are as they are known to be. *)
Lemma known_array_of_sound (known : known_arrays) (vars : env)
    (statics : static_env) (s : simple) (array : known_array) :
  arrays_hold known vars statics ->
  known_array_of known s = Some array ->
  ∃ (vs : list value) (v : value),
    fields_hold vars array vs
    ∧ eval_simple vars s = Some v
    ∧ array_of statics v = Some vs.
Proof.
  intros [Hvars Hsymbols].
  destruct s as [x | sym | c]; simpl; try discriminate; intros Hlookup.
  - destruct (Hvars (x, array) (lookup_in _ _ _ Hlookup))
      as (vs & Hfields & Hx).
    simpl in Hfields, Hx.
    exists vs, (Value_immutable_array vs). auto.
  - destruct (Hsymbols (sym, array) (lookup_in _ _ _ Hlookup))
      as (vs & Hfields & Hsym).
    simpl in Hfields, Hsym.
    exists vs, (Value_symbol sym).
    split; [ exact Hfields | split; [ reflexivity | ] ].
    simpl. rewrite Hsym. reflexivity.
Qed.

(* The primitive, where it has a value, has the one the pass gives. *)
Lemma of_known_array_sound (known : known_arrays) (vars : env)
    (statics : static_env) (p : prim) (args : list simple)
    (result : simple) (v : value) :
  arrays_hold known vars statics ->
  of_known_array known p args = Some result ->
  eval_named statics vars (Prim p args) = Some v ->
  eval_simple vars result = Some v.
Proof.
  intros Hhold Hresult Hd. unfold of_known_array in Hresult.
  destruct p as [u | b | vp | hp | rp]; try discriminate Hresult.
  - (* [Array_length] *)
    destruct u; try discriminate Hresult.
    destruct args as [ | a [ | a' args]]; try discriminate Hresult.
    destruct (known_array_of known a) as [array | ] eqn:Hknown;
      [ | discriminate Hresult ].
    destruct (known_array_of_sound _ _ _ _ _ Hhold Hknown)
      as (vs & array_value & Hfields & Ha & Harray_of).
    simpl in Hd. rewrite Ha in Hd. simpl in Hd.
    rewrite eval_prim_unary in Hd. simpl in Hd.
    rewrite Harray_of in Hd. simpl in Hd.
    unfold array_length_of in Hd. unfold length_of_known in Hresult.
    rewrite (fields_hold_length _ _ _ Hfields) in Hd.
    destruct
      (in_range (int_width Standard_tagged_immediate)
         (Z.of_nat (length array)));
      [ | discriminate Hresult ].
    injection Hresult as <-. exact Hd.
  - (* [Array_load] *)
    destruct b; try discriminate Hresult.
    destruct args as [ | a [ | index [ | arg args]]];
      try discriminate Hresult.
    destruct (known_array_of known a) as [array | ] eqn:Hknown;
      [ | discriminate Hresult ].
    destruct (known_array_of_sound _ _ _ _ _ Hhold Hknown)
      as (vs & array_value & Hfields & Ha & Harray_of).
    unfold element_loaded in Hresult.
    destruct index as [x | sym | c]; try discriminate Hresult.
    destruct c; try discriminate Hresult.
    destruct (0 <=? i)%Z; [ | discriminate Hresult ].
    destruct (nth_error array (Z.to_nat i)) as [[contents | ] | ] eqn:Hnth;
      try discriminate Hresult.
    injection Hresult as ->.
    destruct (fields_hold_nth _ _ _ Hfields _ _ Hnth)
      as (loaded & Hloaded & Hcontents).
    simpl in Hd. rewrite Ha in Hd. simpl in Hd.
    rewrite eval_prim_binary in Hd. simpl in Hd.
    rewrite Harray_of in Hd. simpl in Hd.
    unfold array_loaded, array_index in Hd. simpl in Hd.
    destruct (in_range _ i); [ | discriminate Hd ].
    simpl in Hd.
    destruct ((0 <=? i)%Z && (i <? Z.of_nat (length vs))%Z);
      [ | discriminate Hd ].
    simpl in Hd. rewrite Hloaded in Hd. simpl in Hd.
    destruct (array_element_ok (array_kind_of_load_kind load_kind) loaded);
      [ | discriminate Hd ].
    injection Hd as <-. exact Hcontents.
Qed.

Lemma load_known_array_named_sound (known : known_arrays) (vars : env)
    (statics : static_env) (d : named) (v : value) :
  arrays_hold known vars statics ->
  eval_named statics vars d = Some v ->
  eval_named statics vars (load_known_array_named known d) = Some v.
Proof.
  intros Hhold Hd. unfold load_known_array_named.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    try exact Hd.
  destruct (of_known_array known p args) as [result | ] eqn:Hresult;
    [ | exact Hd ].
  simpl. exact (of_known_array_sound _ _ _ _ _ _ _ Hhold Hresult Hd).
Qed.

(* ======================================================================== *)
(* With the run of the machine                                              *)
(* ======================================================================== *)

(* The machine runs a [Let] with the heap ([exec_named]); the facts
   above are about the value alone ([eval_named]). What the pass records
   is made by a primitive that reads no cell. The primitives that it
   replaces, the length of an array and a load from one, may read a
   cell; but the array is known, so its value is one that a primitive
   made or a symbol bound to one, and no location, and the index of a
   load is a constant: the run has the value that [eval_named] gives
   and leaves the heap and the record
   (FlambdaSemanticsSpec.exec_named_no_location_statement). Of anything
   else the pass records nothing and changes nothing. *)
Lemma arrays_hold_under_let_exec (known : known_arrays) (x : variable)
    (d : named) (vars : env) (statics : static_env) (hp hp' : heap)
    (rd rd' : read_record) (v : value) :
  arrays_hold known vars statics ->
  exec_named statics hp rd vars d = Some (v, hp', rd') ->
  arrays_hold (arrays_under_let known x d) ((x, v) :: vars) statics.
Proof.
  intros Hhold Hd.
  destruct (array_made_by d) as [array | ] eqn:Hmade.
  - apply arrays_hold_under_let; [ exact Hhold | ].
    destruct (array_made_by_spec d array Hmade) as (kind & elements & -> & _).
    exact
      (proj1
         (exec_named_pure statics hp hp' rd rd' vars
            (Prim (Variadic (Make_immutable_array kind)) elements) v
            eq_refl Hd)).
  - unfold arrays_under_let. rewrite Hmade.
    exact (arrays_hold_forget_var known vars statics x v Hhold).
Qed.

(* Where the pass reads, the primitive is not of the heap and none of
   the values of its arguments is a location. *)
Lemma of_known_array_no_location (known : known_arrays) (vars : env)
    (statics : static_env) (p : prim) (args : list simple)
    (result : simple) (vs : list value) :
  arrays_hold known vars statics ->
  of_known_array known p args = Some result ->
  eval_simples vars args = Some vs ->
  (∀ q : heap_primitive, p <> Heap q)
  ∧ ∀ l : location, Value_location l ∉ vs.
Proof.
  intros Hhold Hresult Hvs. unfold of_known_array in Hresult.
  destruct p as [u | b | vp | hp | rp]; try discriminate Hresult.
  - (* [Array_length] *)
    destruct u; try discriminate Hresult.
    destruct args as [ | a [ | a' args]]; try discriminate Hresult.
    destruct (known_array_of known a) as [array | ] eqn:Hknown;
      [ | discriminate Hresult ].
    destruct (known_array_of_sound _ _ _ _ _ Hhold Hknown)
      as (elements & array_value & _ & Ha & Harray_of).
    cbn [eval_simples] in Hvs. rewrite Ha in Hvs. injection Hvs as <-.
    split; [ intros q; discriminate | ].
    intros l [-> | []]. discriminate Harray_of.
  - (* [Array_load] *)
    destruct b; try discriminate Hresult.
    destruct args as [ | a [ | index [ | arg args]]];
      try discriminate Hresult.
    destruct (known_array_of known a) as [array | ] eqn:Hknown;
      [ | discriminate Hresult ].
    destruct (known_array_of_sound _ _ _ _ _ Hhold Hknown)
      as (elements & array_value & _ & Ha & Harray_of).
    unfold element_loaded in Hresult.
    destruct index as [x | sym | c]; try discriminate Hresult.
    cbn [eval_simples eval_simple] in Hvs. rewrite Ha in Hvs.
    injection Hvs as <-.
    split; [ intros q; discriminate | ].
    intros l [-> | [Hl | []]];
      [ discriminate Harray_of | discriminate Hl ].
Qed.

Lemma load_known_array_named_exec (known : known_arrays) (vars : env)
    (statics : static_env) (hp hp' : heap) (rd rd' : read_record)
    (d : named) (v : value) :
  arrays_hold known vars statics ->
  exec_named statics hp rd vars d = Some (v, hp', rd') ->
  exec_named statics hp rd vars (load_known_array_named known d)
  = Some (v, hp', rd').
Proof.
  intros Hhold Hd. unfold load_known_array_named.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    try exact Hd.
  destruct (of_known_array known p args) as [result | ] eqn:Hresult;
    [ | exact Hd ].
  destruct (eval_simples vars args) as [vs | ] eqn:Hvs.
  2: { rewrite exec_named_prim, Hvs in Hd. discriminate Hd. }
  destruct
    (of_known_array_no_location known vars statics p args result vs Hhold
       Hresult Hvs)
    as [Hnot_heap Hno_location].
  rewrite
    (exec_named_no_location statics hp rd vars p args vs Hnot_heap Hvs
       Hno_location)
    in Hd.
  destruct (eval_named statics vars (Prim p args)) as [w | ] eqn:Heval;
    [ | discriminate Hd ].
  injection Hd as <- <- <-.
  cbn [exec_named eval_named].
  rewrite
    (of_known_array_sound known vars statics p args result w Hhold Hresult
       Heval).
  reflexivity.
Qed.

(* ======================================================================== *)
(* The pass                                                                 *)
(* ======================================================================== *)

(* At a [Let], the primitive has the value that the pass gives or none;
   then the parts under binders, by the induction, with what is known
   there. *)
Fixpoint load_arrays_knowing_refines (e : expr) :
  ∀ (known : known_arrays) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record),
    arrays_hold known vars statics ->
    refines (State e vars conts statics hp rd)
      (State (load_arrays_knowing known e) vars conts statics hp rd)
with load_arrays_in_handler_refines (h : continuation_handler) :
  ∀ (known : known_arrays) (k : continuation) (body : expr) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record),
    arrays_hold known vars statics ->
    refines (State (Let_cont_non_recursive k h body) vars conts statics hp rd)
      (State
         (Let_cont_non_recursive k (load_arrays_in_handler known h) body)
         vars conts statics hp rd).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap
         | ac | scrutinee arms | ];
      intros known vars conts statics hp rd Hhold; simpl.
    + (* Let: the machine runs [d] with the heap; the pass runs the
         same or what is known of the array, and leaves the same heap
         and record. *)
      apply let_refines_step_forward.
      * intros v hp' rd' Hd.
        exact
          (load_known_array_named_exec known vars statics hp hp' rd rd' d v
             Hhold Hd).
      * intros v hp' rd' Hd _. apply load_arrays_knowing_refines.
        exact
          (arrays_hold_under_let_exec known x d vars statics hp hp' rd rd' v
             Hhold Hd).
    + (* Let_static *)
      apply let_static_refines_step.
      destruct (eval_static_bindings vars bindings statics)
        as [extended | ] eqn:Hbindings; [ | exact I ].
      apply load_arrays_knowing_refines.
      exact (arrays_hold_under_statics known vars statics bindings extended
        Hhold Hbindings).
    + (* Let_cont_non_recursive *)
      apply (refines_trans _
        (State
           (Let_cont_non_recursive k (load_arrays_in_handler known h) body)
           vars conts statics hp rd)).
      * apply load_arrays_in_handler_refines, Hhold.
      * apply let_cont_non_recursive_refines_step,
          load_arrays_knowing_refines, Hhold.
    + (* Let_cont_recursive *)
      apply let_cont_recursive_refines_step; [ reflexivity | intros _ ].
      apply load_arrays_knowing_refines, Hhold.
    + apply refines_refl.
    + apply refines_refl.
    + apply refines_refl.
    + apply refines_refl.
  - destruct h as [params metadata handler].
    intros known k body vars conts statics hp rd Hhold. simpl.
    apply (let_cont_handler_refines_where (array_symbols_hold known vars)
      (array_symbols_hold_extend known vars));
      [ exact (proj2 Hhold) | ].
    intros vs handler_vars conts' statics' hp' rd' Hsymbols _ Hbind.
    apply load_arrays_knowing_refines.
    apply (arrays_hold_bind_params statics' params known vs vars
      handler_vars);
      [ split; [ exact (proj1 Hhold) | exact Hsymbols ] | exact Hbind ].
Qed.

Theorem load_known_arrays_refines : load_known_arrays_refines_statement.
Proof.
  intros e vars conts statics hp rd. apply load_arrays_knowing_refines.
  split; intros entry [].
Qed.

(* ======================================================================== *)
(* Renaming                                                                 *)
(* ======================================================================== *)

(* A map whose keys are renamed: a renaming is injective, so a variable
   is found where its image is. *)
Lemma lookup_renamed_array_vars {A B : Type} (r : renaming) (g : A -> B)
    (x : variable) (l : list (variable * A)) :
  lookup (rename_variable r x)
    (map (fun entry => (rename_variable r (fst entry), g (snd entry))) l)
  = option_map g (lookup x l).
Proof.
  induction l as [ | [y a] l IH]; simpl; [ reflexivity | ].
  rewrite rename_variable_eqb.
  destruct (Pos.eqb x y); [ reflexivity | exact IH ].
Qed.

Lemma filter_map_array_vars {A B : Type} (keep' : B -> bool) (g : A -> B)
    (keep : A -> bool) (l : list A) :
  (∀ a : A, keep' (g a) = keep a) ->
  filter keep' (map g l) = map g (filter keep l).
Proof.
  intros Hkeep. induction l as [ | a l IH]; simpl; [ reflexivity | ].
  rewrite Hkeep, IH. destruct (keep a); reflexivity.
Qed.

Lemma forget_array_field_equivariant (r : renaming) (x : variable)
    (field : option simple) :
  forget_field (rename_variable r x) (option_map (rename_simple r) field)
  = option_map (rename_simple r) (forget_field x field).
Proof.
  destruct field as [s | ]; simpl; [ | reflexivity ].
  rewrite var_in_simple_equivariant. destruct (var_in_simple x s); reflexivity.
Qed.

Lemma forget_in_array_equivariant (r : renaming) (x : variable)
    (array : known_array) :
  forget_in_array (rename_variable r x) (rename_known_array r array)
  = rename_known_array r (forget_in_array x array).
Proof.
  unfold forget_in_array, rename_known_array. rewrite !map_map.
  apply map_ext. intros field. apply forget_array_field_equivariant.
Qed.

Lemma forget_array_var_equivariant (r : renaming) (x : variable)
    (known : known_arrays) :
  forget_array_var (rename_variable r x) (rename_known_arrays r known)
  = rename_known_arrays r (forget_array_var x known).
Proof.
  unfold forget_array_var, rename_known_arrays, map_values. simpl. f_equal.
  - rewrite
      (filter_map_array_vars _ _
         (fun entry => negb (Pos.eqb x (fst entry))))
      by (intros entry; simpl; rewrite rename_variable_eqb; reflexivity).
    rewrite !map_map. apply map_ext. intros entry. simpl.
    rewrite forget_in_array_equivariant. reflexivity.
  - rewrite !map_map. apply map_ext. intros [sym array]. simpl.
    rewrite forget_in_array_equivariant. reflexivity.
Qed.

Lemma forget_array_vars_equivariant (r : renaming) (xs : list variable) :
  ∀ known : known_arrays,
    forget_array_vars (map (rename_variable r) xs)
      (rename_known_arrays r known)
    = rename_known_arrays r (forget_array_vars xs known).
Proof.
  induction xs as [ | x xs IH]; intros known; simpl; [ reflexivity | ].
  rewrite forget_array_var_equivariant. apply IH.
Qed.

Lemma array_made_by_equivariant (r : renaming) (d : named) :
  array_made_by (rename_named r d)
  = option_map (rename_known_array r) (array_made_by d).
Proof.
  destruct d as [s | p args | | | | | ]; try reflexivity.
  destruct p as [p | p | [kind | kind] | p | rp]; try reflexivity.
  simpl. unfold rename_known_array. rewrite !map_map. reflexivity.
Qed.

Lemma arrays_under_let_equivariant (r : renaming) (known : known_arrays)
    (x : variable) (d : named) :
  arrays_under_let (rename_known_arrays r known) (rename_variable r x)
    (rename_named r d)
  = rename_known_arrays r (arrays_under_let known x d).
Proof.
  unfold arrays_under_let.
  rewrite forget_array_var_equivariant, array_made_by_equivariant.
  destruct (array_made_by d) as [array | ]; cbn [option_map];
    [ | reflexivity ].
  rewrite forget_in_array_equivariant. reflexivity.
Qed.

Lemma static_array_made_renamed (r : renaming) (binding : static_binding) :
  static_array_made (rename_static_binding r binding)
  = map_values (rename_known_array r) (static_array_made binding).
Proof.
  destruct binding as [id c | id | sym const]; try reflexivity.
  destruct const; try reflexivity.
  simpl. unfold rename_known_array. rewrite !map_map. reflexivity.
Qed.

Lemma arrays_under_statics_equivariant (r : renaming)
    (known : known_arrays) (bindings : list static_binding) :
  arrays_under_statics (rename_known_arrays r known)
    (map (rename_static_binding r) bindings)
  = rename_known_arrays r (arrays_under_statics known bindings).
Proof.
  unfold arrays_under_statics, rename_known_arrays, map_values. simpl.
  f_equal. rewrite map_app. f_equal.
  induction bindings as [ | binding bindings IH]; simpl; [ reflexivity | ].
  rewrite static_array_made_renamed, map_app, IH. reflexivity.
Qed.

Lemma known_array_of_equivariant (r : renaming) (known : known_arrays)
    (s : simple) :
  known_array_of (rename_known_arrays r known) (rename_simple r s)
  = option_map (rename_known_array r) (known_array_of known s).
Proof.
  destruct s as [x | sym | c]; simpl;
    [ apply lookup_renamed_array_vars | apply lookup_map_values
    | reflexivity ].
Qed.

Lemma length_of_known_equivariant (r : renaming) (array : known_array) :
  length_of_known (rename_known_array r array)
  = option_map (rename_simple r) (length_of_known array).
Proof.
  unfold length_of_known, rename_known_array. rewrite length_map.
  destruct
    (in_range (int_width Standard_tagged_immediate)
       (Z.of_nat (length array)));
    reflexivity.
Qed.

Lemma element_loaded_equivariant (r : renaming) (array : known_array)
    (index : simple) :
  element_loaded (rename_known_array r array) (rename_simple r index)
  = option_map (rename_simple r) (element_loaded array index).
Proof.
  destruct index as [x | sym | c]; try reflexivity.
  destruct c; try reflexivity. simpl.
  destruct (0 <=? i)%Z; [ | reflexivity ].
  unfold rename_known_array. rewrite nth_error_map.
  destruct (nth_error array (Z.to_nat i)) as [[contents | ] | ];
    reflexivity.
Qed.

Lemma of_known_array_equivariant (r : renaming) (known : known_arrays)
    (p : prim) (args : list simple) :
  of_known_array (rename_known_arrays r known) p
    (map (rename_simple r) args)
  = option_map (rename_simple r) (of_known_array known p args).
Proof.
  destruct p as [u | b | vp | hp | rp]; try reflexivity.
  - destruct u; try reflexivity.
    destruct args as [ | a [ | a' args]]; try reflexivity.
    cbn [of_known_array map]. rewrite known_array_of_equivariant.
    destruct (known_array_of known a) as [array | ]; cbn [option_map];
      [ | reflexivity ].
    apply length_of_known_equivariant.
  - destruct b; try reflexivity.
    destruct args as [ | a [ | index [ | arg args]]]; try reflexivity.
    cbn [of_known_array map]. rewrite known_array_of_equivariant.
    destruct (known_array_of known a) as [array | ]; cbn [option_map];
      [ | reflexivity ].
    apply element_loaded_equivariant.
Qed.

Lemma load_known_array_named_equivariant (r : renaming)
    (known : known_arrays) (d : named) :
  load_known_array_named (rename_known_arrays r known) (rename_named r d)
  = rename_named r (load_known_array_named known d).
Proof.
  destruct d as [s | p args | | | | | ]; try reflexivity.
  cbn [rename_named load_known_array_named].
  rewrite of_known_array_equivariant.
  destruct (of_known_array known p args); reflexivity.
Qed.

Fixpoint load_arrays_knowing_equivariant_aux (r : renaming) (e : expr) :
  ∀ known : known_arrays,
    load_arrays_knowing (rename_known_arrays r known) (rename_expr r e)
    = rename_expr r (load_arrays_knowing known e)
with load_arrays_in_handler_equivariant (r : renaming)
    (h : continuation_handler) :
  ∀ known : known_arrays,
    load_arrays_in_handler (rename_known_arrays r known)
      (rename_handler r h)
    = rename_handler r (load_arrays_in_handler known h).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros known; try reflexivity;
      cbn [rename_expr load_arrays_knowing].
    + rewrite load_known_array_named_equivariant,
        arrays_under_let_equivariant, load_arrays_knowing_equivariant_aux.
      reflexivity.
    + rewrite arrays_under_statics_equivariant,
        load_arrays_knowing_equivariant_aux.
      reflexivity.
    + rewrite load_arrays_in_handler_equivariant,
        load_arrays_knowing_equivariant_aux.
      reflexivity.
    + rewrite load_arrays_knowing_equivariant_aux. reflexivity.
  - destruct h as [params metadata handler]. intros known.
    cbn [rename_handler load_arrays_in_handler].
    rewrite forget_array_vars_equivariant,
      load_arrays_knowing_equivariant_aux.
    reflexivity.
Qed.

Theorem load_arrays_knowing_equivariant :
  load_arrays_knowing_equivariant_statement.
Proof. intros r known e. apply load_arrays_knowing_equivariant_aux. Qed.

Theorem load_known_arrays_equivariant :
  load_known_arrays_equivariant_statement.
Proof.
  intros r e. exact (load_arrays_knowing_equivariant_aux r e no_arrays).
Qed.

(* ======================================================================== *)
(* What the pass makes of some terms                                        *)
(* ======================================================================== *)

(* As checks of the definitions against what Simplify does. *)

Module Examples.

Definition jump (k : continuation) (args : list simple) : expr :=
  Apply_cont (Apply_cont_expr k args None).

(* [let v3 = Make_immutable_array immediates (v1, v2) in body]. *)
Definition made (body : expr) : expr :=
  Let 3%positive Value
    (Prim (Variadic (Make_immutable_array Array_immediates))
       [Var 1%positive; Var 2%positive])
    body.

Definition length_of (array : simple) : named :=
  Prim (Unary (Array_length (Array_kind Array_immediates))) [array].

Definition load (array index : simple) : named :=
  Prim (Binary (Array_load Array_immediates Load_immediates Immutable))
    [array; index].

Definition reads (d : named) : expr :=
  Let 4%positive Value d (jump 5%positive [Var 4%positive]).

(* The length of an array that a [Let] has made is the number of its
   arguments. *)
Example length_of_a_made_array :
  load_known_arrays (made (reads (length_of (Var 3%positive))))
  = made (reads (Simple (Const (Tagged_immediate 2)))).
Proof. reflexivity. Qed.

(* A load at a constant index within the bounds is the argument at that
   index. *)
Example load_of_a_made_array :
  load_known_arrays
    (made (reads (load (Var 3%positive) (Const (Tagged_immediate 1)))))
  = made (reads (Simple (Var 2%positive))).
Proof. reflexivity. Qed.

(* A load at a constant index that the array has not got is left. *)
Example load_out_of_bounds_left :
  load_known_arrays
    (made (reads (load (Var 3%positive) (Const (Tagged_immediate 2)))))
  = made (reads (load (Var 3%positive) (Const (Tagged_immediate 2)))).
Proof. reflexivity. Qed.

Example load_at_a_negative_index_left :
  load_known_arrays
    (made (reads (load (Var 3%positive) (Const (Tagged_immediate (-1))))))
  = made (reads (load (Var 3%positive) (Const (Tagged_immediate (-1))))).
Proof. reflexivity. Qed.

(* A load at an index that is a variable is left. *)
Example load_at_a_variable_left :
  load_known_arrays
    (made (reads (load (Var 3%positive) (Var 1%positive))))
  = made (reads (load (Var 3%positive) (Var 1%positive))).
Proof. reflexivity. Qed.

(* Once the variable that an element held is bound again, what that
   element holds is not known; the length is, and the other element. *)
Definition rebound (body : expr) : expr :=
  made
    (Let 2%positive Value (Simple (Const (Tagged_immediate 9))) body).

Example load_of_a_rebound_element_left :
  load_known_arrays
    (rebound (reads (load (Var 3%positive) (Const (Tagged_immediate 1)))))
  = rebound (reads (load (Var 3%positive) (Const (Tagged_immediate 1)))).
Proof. reflexivity. Qed.

Example length_after_a_rebinding :
  load_known_arrays (rebound (reads (length_of (Var 3%positive))))
  = rebound (reads (Simple (Const (Tagged_immediate 2)))).
Proof. reflexivity. Qed.

Example load_of_the_other_element :
  load_known_arrays
    (rebound (reads (load (Var 3%positive) (Const (Tagged_immediate 0)))))
  = rebound (reads (Simple (Var 1%positive))).
Proof. reflexivity. Qed.

(* A symbol bound to a static immutable array, in the body of its
   [Let_static]. *)
Definition bound (body : expr) : expr :=
  Let_static
    [Bind_symbol 7%positive
       (Static_const_immutable_value_array
          [Const (Tagged_immediate 3); Var 1%positive])]
    body.

Example length_of_a_static_array :
  load_known_arrays (bound (reads (length_of (Symbol 7%positive))))
  = bound (reads (Simple (Const (Tagged_immediate 2)))).
Proof. reflexivity. Qed.

Example load_of_a_static_array :
  load_known_arrays
    (bound (reads (load (Symbol 7%positive) (Const (Tagged_immediate 0)))))
  = bound (reads (Simple (Const (Tagged_immediate 3)))).
Proof. reflexivity. Qed.

(* A symbol that no [Let_static] around binds to an array is not known. *)
Example length_of_an_unknown_symbol_left :
  load_known_arrays (reads (length_of (Symbol 7%positive)))
  = reads (length_of (Symbol 7%positive)).
Proof. reflexivity. Qed.

(* The handler of a non-recursive continuation knows what is known
   where it is defined, less its parameters. *)
Definition in_handler (params : list variable) (handler : expr) : expr :=
  made
    (Let_cont_non_recursive 6%positive
       (Handler params (make_handler_metadata (map (fun _ => Value) params))
          handler)
       (jump 6%positive (map Var params))).

Example length_in_a_handler :
  load_known_arrays (in_handler [] (reads (length_of (Var 3%positive))))
  = in_handler [] (reads (Simple (Const (Tagged_immediate 2)))).
Proof. reflexivity. Qed.

Example array_rebound_by_a_parameter :
  load_known_arrays
    (in_handler [3%positive] (reads (length_of (Var 3%positive))))
  = in_handler [3%positive] (reads (length_of (Var 3%positive))).
Proof. reflexivity. Qed.

End Examples.

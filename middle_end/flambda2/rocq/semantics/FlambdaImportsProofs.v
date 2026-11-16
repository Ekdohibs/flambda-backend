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

(* Proofs of the statements of FlambdaImportsSpec.v. Nothing here needs
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaSemanticsProofs FlambdaPrimitiveProofs
  FlambdaUnusedStaticsSpec FlambdaImportsSpec.
Import ListNotations.

Theorem expr_refines_unit_refines : expr_refines_unit_refines_statement.
Proof. intros imports source target Hrefines. apply Hrefines. Qed.

(* ======================================================================== *)
(* The static environment of the imports                                    *)
(* ======================================================================== *)

(* Three facts for the proofs that start a unit from its imports: there
   is no code in that environment, what it binds a symbol to is something
   imported, and no name occurs in it. *)
Lemma imports_statics_code (imports : list (symbol * import_class)) :
  static_code (imports_statics imports) = [].
Proof. reflexivity. Qed.

Lemma imports_statics_imported (imports : list (symbol * import_class))
    (s : symbol) (allocated : statically_allocated_value) :
  lookup s (static_values (imports_statics imports)) = Some allocated ->
  ∃ c : import_class, allocated = Static_imported c.
Proof.
  unfold imports_statics. cbn [static_values].
  induction imports as [ | [s' c] imports IH]; cbn [map lookup];
    [ discriminate | ].
  destruct (Pos.eqb s s'); [ | exact IH ].
  intros [= <-]. exists c. reflexivity.
Qed.

Lemma imports_statics_free_of (D : dropped)
    (imports : list (symbol * import_class)) :
  envs_free_of D [] [] (imports_statics imports) = true.
Proof.
  unfold envs_free_of, statics_free_of, imports_statics. simpl.
  induction imports as [ | [s c] imports IH]; [ reflexivity | exact IH ].
Qed.

(* ======================================================================== *)
(* A prefix of static bindings                                              *)
(* ======================================================================== *)

Theorem prefix_refines : prefix_refines_statement.
Proof.
  intros prefix statics statics' hp rd source target Heval.
  assert (Hsource :
    step (State (Let_static prefix source) [] [] statics hp rd)
    = Step (State source [] [] statics' hp rd))
    by (rewrite step_let_static, Heval; reflexivity).
  assert (Htarget :
    step (State (Let_static prefix target) [] [] statics hp rd)
    = Step (State target [] [] statics' hp rd))
    by (rewrite step_let_static, Heval; reflexivity).
  split; intros Hrefines.
  - apply (refines_trans _ _ _
      (equivalent_refines _ _
         (equivalent_sym _ _ (step_equivalent _ _ Hsource)))).
    apply (refines_trans _ _ _ Hrefines).
    apply equivalent_refines, step_equivalent, Htarget.
  - exact (refines_step _ _ _ _ Hsource Htarget Hrefines).
Qed.

(* ======================================================================== *)
(* What a primitive does with an imported symbol                            *)
(* ======================================================================== *)

Lemma identity_not_allocated (statics : static_env) (v : value)
    (s : symbol) :
  identity_of statics v = Some s -> is_allocated statics v = false.
Proof.
  intros Hs. destruct (is_allocated statics v) eqn:Hallocated;
    [ | reflexivity ].
  unfold identity_of in Hs.
  rewrite (is_allocated_not_imported statics v Hallocated) in Hs.
  discriminate.
Qed.

Lemma identity_is_imported (statics : static_env) (v : value) (s : symbol) :
  identity_of statics v = Some s -> is_imported statics v = true.
Proof.
  intros Hs. apply identity_of_imported in Hs. unfold is_imported.
  rewrite Hs. reflexivity.
Qed.

Theorem phys_equal_identities : phys_equal_identities_statement.
Proof.
  intros statics s1 s2 Hs1 Hs2. unfold eval_phys_equal.
  rewrite !int_of_value_symbol, Hs1, Hs2. reflexivity.
Qed.

Theorem phys_equal_imported_immediate
  : phys_equal_imported_immediate_statement.
Proof.
  intros statics s v Hs Hv. unfold eval_phys_equal.
  unfold is_tagged_immediate in Hv.
  rewrite !int_of_value_symbol, Hs.
  destruct (int_of_value Standard_tagged_immediate v) as [i | ];
    [ | discriminate ].
  rewrite orb_true_r. split; reflexivity.
Qed.

Theorem phys_equal_identity_other : phys_equal_identity_other_statement.
Proof.
  intros statics s v Hs Hv. unfold eval_phys_equal.
  rewrite !int_of_value_symbol, Hs.
  assert (Hnone : identity_of statics v = None).
  { destruct Hv as [Hv | [_ Hv]]; [ | exact Hv ].
    unfold identity_of.
    rewrite (is_allocated_not_imported statics v Hv). reflexivity. }
  assert
    (Hother : is_allocated statics v || is_imported_pointer statics v = true).
  { destruct Hv as [Hv | [Hv _]]; rewrite Hv;
      [ reflexivity | apply orb_true_r ]. }
  assert (Hint : int_of_value Standard_tagged_immediate v = None).
  { destruct (int_of_value Standard_tagged_immediate v) as [i | ] eqn:Hint;
      [ | reflexivity ].
    exfalso. destruct Hv as [Hv | [Hv _]].
    - destruct v; simpl in Hint, Hv; discriminate.
    - unfold is_imported_pointer in Hv.
      destruct v; simpl in Hint, Hv; discriminate. }
  rewrite Hint, Hnone, Hother. split; reflexivity.
Qed.

Theorem phys_equal_data_import : phys_equal_data_import_statement.
Proof.
  intros statics s v Hs Hv Hnone. unfold eval_phys_equal.
  unfold is_tagged_immediate in Hv.
  assert (Hdata : identity_of statics (Value_symbol s) = None).
  { unfold identity_of. rewrite Hs. reflexivity. }
  rewrite !int_of_value_symbol, Hdata, Hnone.
  destruct (int_of_value Standard_tagged_immediate v) as [i | ];
    [ discriminate | ].
  split; reflexivity.
Qed.

Theorem phys_equal_unknown_import : phys_equal_unknown_import_statement.
Proof.
  intros statics s v Hs. unfold eval_phys_equal.
  assert (Hidentity : identity_of statics (Value_symbol s) = None).
  { unfold identity_of. rewrite Hs. reflexivity. }
  assert (Hpointer : is_imported_pointer statics (Value_symbol s) = false).
  { unfold is_imported_pointer. rewrite Hs. reflexivity. }
  assert (Hallocated : is_allocated statics (Value_symbol s) = false).
  { destruct (is_allocated statics (Value_symbol s)) eqn:Hallocated;
      [ | reflexivity ].
    rewrite (is_allocated_not_imported statics _ Hallocated) in Hs.
    discriminate. }
  rewrite !int_of_value_symbol, Hidentity, Hpointer, Hallocated.
  destruct (int_of_value Standard_tagged_immediate v) as [i | ];
    [ split; reflexivity | ].
  destruct (identity_of statics v); split; reflexivity.
Qed.

Theorem imported_pointer : imported_pointer_statement.
Proof.
  split.
  - intros statics v Hnot. unfold is_imported_pointer, is_imported.
    destruct (imported_of statics v) as [[s [ | | ]] | ]; try reflexivity.
    exfalso. exact (Hnot s eq_refl).
  - intros statics v s Hs. unfold is_imported_pointer, is_imported.
    rewrite Hs. split; reflexivity.
Qed.

Lemma pointer_is_imported (statics : static_env) (v : value) :
  no_import_unknown statics ->
  is_imported_pointer statics v = is_imported statics v.
Proof.
  intros Hno. apply (proj1 imported_pointer).
  intros s Hs. apply imported_of_symbol in Hs as [_ Hs].
  exact (Hno s Hs).
Qed.

Theorem phys_equal_two_classes : phys_equal_two_classes_statement.
Proof.
  intros statics v1 v2 Hno. split.
  - unfold eval_phys_equal, eval_phys_equal_two_classes.
    rewrite !(pointer_is_imported statics _ Hno). reflexivity.
  - unfold eval_phys_equal_location, eval_phys_equal_location_two_classes.
    destruct v1, v2; try reflexivity;
      rewrite !(pointer_is_imported statics _ Hno); reflexivity.
Qed.

(* A static binding binds no import. *)
Lemma eval_static_bindings_no_import_unknown (vars : env)
    (bindings : list static_binding) :
  ∀ statics extended : static_env,
    eval_static_bindings vars bindings statics = Some extended ->
    no_import_unknown statics -> no_import_unknown extended.
Proof.
  induction bindings as [ | binding bindings IH]; intros statics extended;
    simpl.
  - intros [= <-] Hno. exact Hno.
  - destruct binding as [id c | id | sym const].
    + destruct (lookup id (static_code statics)); [ discriminate | ].
      intros Heval Hno. exact (IH _ _ Heval Hno).
    + apply IH.
    + destruct (lookup sym (static_values statics)); [ discriminate | ].
      destruct (eval_static_const vars const) as [allocated | ] eqn:Hconst;
        simpl; [ | discriminate ].
      intros Heval Hno. apply (IH _ _ Heval).
      intros s. simpl. destruct (Pos.eqb s sym).
      * intros [= ->]. destruct const; simpl in Hconst;
          repeat
            match type of Hconst with
            | context [match ?x with _ => _ end] => destruct x
            end;
          discriminate.
      * apply Hno.
Qed.

Lemma step_no_import_unknown (s s' : state) :
  step s = Step s' ->
  no_import_unknown (state_statics s) ->
  no_import_unknown (state_statics s').
Proof.
  intros Hstep. rewrite (state_eta s) in Hstep.
  set (vars := state_vars s) in *. set (conts := state_conts s) in *.
  set (statics := state_statics s) in *.
  destruct (state_expr s)
    as [x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    autorewrite with step in Hstep.
  - destruct (exec_named _ _ rd_ _ d) as [[[v hp'] rd'] | ]; [ | discriminate ].
    destruct (kind_eqb (kind_of_value v) kind); [ | discriminate ].
    injection Hstep as <-. exact (fun H => H).
  - destruct (eval_static_bindings vars bindings statics) as [extended | ]
      eqn:Heval; [ | discriminate ].
    injection Hstep as <-.
    exact
      (eval_static_bindings_no_import_unknown vars bindings statics extended
         Heval).
  - injection Hstep as <-. exact (fun H => H).
  - destruct (handlers_bind_statics handlers); [ discriminate | ].
    injection Hstep as <-. exact (fun H => H).
  - destruct (eval_simple vars (ap_callee ap)) as [callee | ];
      [ | discriminate ].
    destruct (eval_simples vars (ap_args ap)) as [vs | ]; [ | discriminate ].
    destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ];
      [ | discriminate ].
    rewrite (call_function_statics _ _ _ _ _ _ _ _ _ rd_ _ Hstep).
    exact (fun H => H).
  - destruct (eval_simples vars (ac_args ac)) as [vs | ]; [ | discriminate ].
    unfold apply_continuation in Hstep.
    rewrite (apply_cont_value_statics _ _ _ _ rd_ _ Hstep).
    exact (fun H => H).
  - destruct (eval_simple vars scrutinee) as [v | ]; [ | discriminate ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | discriminate ].
    destruct (find_arm i arms); [ | discriminate ].
    injection Hstep as <-. exact (fun H => H).
  - discriminate.
Qed.

Lemma imports_statics_no_import_unknown
    (imports : list (symbol * import_class)) :
  (∀ (s : symbol) (c : import_class),
    (s, c) ∈ imports -> c <> Import_unknown) ->
  no_import_unknown (imports_statics imports).
Proof.
  intros Hall s. unfold imports_statics. simpl.
  induction imports as [ | [s0 c0] imports IH]; simpl; [ discriminate | ].
  destruct (Pos.eqb s s0).
  - intros [= ->].
    exact (Hall s0 Import_unknown (or_introl eq_refl) eq_refl).
  - apply IH. intros s1 c1 Hin. exact (Hall s1 c1 (or_intror Hin)).
Qed.

Theorem unit_two_classes : unit_two_classes_statement.
Proof.
  assert
    (Hrun :
      ∀ (fuel : nat) (s reached : state),
        run fuel s = Step reached ->
        no_import_unknown (state_statics s) ->
        no_import_unknown (state_statics reached)).
  { intros fuel. induction fuel as [ | fuel IH]; intros s reached; simpl.
    - intros [= <-] Hno. exact Hno.
    - destruct (step s) as [next | k vs | ] eqn:Hstep; try discriminate.
      intros Hreached Hno.
      exact
        (IH next reached Hreached (step_no_import_unknown s next Hstep Hno)). }
  split; [ exact Hrun | ].
  intros imports e fuel reached Hall Hreached.
  exact
    (Hrun fuel _ reached Hreached
       (imports_statics_no_import_unknown imports Hall)).
Qed.

Theorem imported_not_read : imported_not_read_statement.
Proof.
  split.
  - intros statics p v result Himported.
    unfold is_imported in Himported.
    destruct (imported_of statics v) as [[s c] | ] eqn:Hs;
      [ | discriminate ].
    apply imported_of_symbol in Hs as [-> Hs].
    destruct p as [[tag size field_kind] mut field | variant_only | | |
                  length_kind | | kind | kind op | src dst | | kind | kind | |
                  | reinterpret];
      cbn [eval_unary_primitive];
      unfold eval_block_load, eval_is_int, eval_is_null, is_tagged_immediate,
        is_allocated, eval_reinterpret_64_bit_word;
      try destruct reinterpret;
      rewrite ?int_of_value_symbol;
      cbn [block_of closure_of boxed_number_of string_of array_of
           bound_if_symbol partial_application_of kind_of_value];
      rewrite ?Hs; simpl;
      try discriminate.
    + destruct variant_only; discriminate.
    + destruct kind; simpl; try discriminate.
      intros [= <-]. split; reflexivity.
  - intros statics p v1 v2 result Himported.
    destruct p as [array_kind load_kind mut | width | op | kind op | kind op
                  | kind behaviour];
      cbn [eval_binary_primitive]; [ | | intros _; eauto | | | ];
      intros Heval; exfalso;
      destruct Himported as [Himported | Himported];
      unfold is_imported in Himported;
      (destruct (imported_of statics _) as [[s c] | ] eqn:Hs;
         [ | discriminate ]);
      apply imported_of_symbol in Hs as [-> Hs];
      revert Heval;
      unfold eval_string_load, array_loaded, array_index;
      rewrite ?int_of_value_symbol;
      cbn [string_of array_of]; rewrite ?Hs;
      try discriminate.
    + destruct (array_of statics v1); discriminate.
    + destruct (string_of statics v1); discriminate.
    + destruct (int_of_value kind v1); discriminate.
    + destruct (int_of_value kind v1); discriminate.
    + destruct (int_of_value kind v1); discriminate.
Qed.

(* ======================================================================== *)
(* The examples                                                             *)
(* ======================================================================== *)

Theorem catches_by_name : catches_by_name_statement.
Proof.
  unfold catches_by_name_statement. split; [ | split ].
  - exists 10%nat; eexists. vm_compute. reflexivity.
  - exists 10%nat; eexists. vm_compute. reflexivity.
  - exists 12%nat; eexists. vm_compute. reflexivity.
Qed.

Theorem catches_with_argument : catches_with_argument_statement.
Proof.
  unfold catches_with_argument_statement. split.
  - exists 12%nat; eexists. vm_compute. reflexivity.
  - exists 12%nat. vm_compute. reflexivity.
Qed.

Theorem unknown_import : unknown_import_statement.
Proof.
  unfold unknown_import_statement. split; [ | split ].
  - exists 10%nat; eexists. vm_compute. reflexivity.
  - exists 10%nat. vm_compute. reflexivity.
  - exists 12%nat. vm_compute. reflexivity.
Qed.

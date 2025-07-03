import streamlit as st
import matplotlib.pyplot as plt
import torch
import time

from deep_qlearning import train_dqn
from test_agent import evaluate_agent

if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'scores' not in st.session_state:
    st.session_state.scores = []
if 'avg_scores' not in st.session_state:
    st.session_state.avg_scores = []

st.title("Taxi DQN")
st.sidebar.header("Hyperparameters")
# min_value=None, max_value=None, value="min", step=None
lr = st.sidebar.number_input("Learning Rate", 0.0001, 0.01, 0.001, format='%.4f')
gamma = st.sidebar.number_input("Discount Factor (Gamma)", 0.80, 0.99, 0.95)
epsilon = st.sidebar.number_input("Initial Epsilon", 0.0, 1.0, 1.0)
epsilon_min = st.sidebar.number_input("Minimum Epsilon", 0.01, 0.5, 0.01)
epsilon_decay = st.sidebar.number_input("Epsilon Decay", 0.80, 0.999, 0.995)
episodes = st.sidebar.number_input("Episodes", 500, 20000, 1000)
batch_size = st.sidebar.number_input("Batch Size", 16, 256, 64)

# Button to start training
st.markdown("<h2 style='text-align: center;'>Train agent</h2>", unsafe_allow_html=True)

# st.subheader('Train agent')
if st.button("Start Training", use_container_width=True):
    st.session_state.agent, st.session_state.scores, st.session_state.avg_scores = train_dqn(
        lr=lr,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        episodes=episodes,
        batch_size=batch_size,
    )

    fig = plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(st.session_state.scores)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    plt.subplot(1, 2, 2)
    plt.plot(st.session_state.avg_scores)
    plt.title('Average Reward')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    
    plt.tight_layout()
    st.pyplot(fig)

if st.session_state.agent is not None:
    st.info("✅ Agent Trained")
    
    # Show training results if available
    if st.session_state.scores:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Episodes Completed", len(st.session_state.scores))
        with col2:
            st.metric("Final Average Score", f"{st.session_state.avg_scores[-1]:.2f}" if st.session_state.avg_scores else "N/A")
else:
    st.info("ℹ️ No trained agent available. Run training first.")

# load_from_file = st.checkbox("Load agent from file")
# # Conditional text input
# if load_from_file:
#     file_name = st.text_input("Enter filename to load agent")

if st.session_state.agent is not None:
    st.divider()
    st.markdown("<h2 style='text-align: center;'>Evaluate agent</h2>", unsafe_allow_html=True)
    # st.subheader('Evaluate agent')
    n_test = st.number_input("Number of testing steps", 10, 20000, 200)

    if st.button("Evaluate agent", use_container_width=True):
        res = evaluate_agent(st.session_state.agent, n_test)

        col3, col4, col5 = st.columns(3)
        with col3:
            st.metric("Average Reward", f"{res['avg_reward']:.2f}")
        with col4:
            st.metric("Average Steps", f"{res['avg_steps']:.2f}")
        with col5:
            st.metric("Success Rate", f"{res['success_rate'] * 100:.1f}%")

    st.divider()
    st.markdown("<h2 style='text-align: center;'>Save agent</h2>", unsafe_allow_html=True)

    # st.subheader('Save agent')
    model_name = st.text_input("Agent name to save")
    # if st.button("Download model", use_container_width=True):
    #     if model_name:
    #         # st.write("Debug Information:")
    #         # st.write(f"Agent type: {type(st.session_state.agent)}")
    #         # st.write(f"Agent has save method: {hasattr(st.session_state.agent, 'save')}")
    #         # st.write(f"Agent methods: {[method for method in dir(st.session_state.agent) if not method.startswith('_')]}")
            
    #         try:
    #             print(f'saving model {model_name}')
    #             st.session_state.agent.save(f'dqn_agents\\{model_name}.pth')
    #             st.success(f"Model saved as {model_name}.pth")
    #         except Exception as e:
    #             st.error(f"Error saving model: {str(e)}")
    #     else:
    #         st.warning("Please enter a model name")


    if st.button("Prepare Download", use_container_width=True):
        if model_name:
            try:
                # Create a BytesIO buffer to save the model
                buffer = io.BytesIO()
                
                # Save the model state dict to the buffer
                torch.save(st.session_state.agent.q_network.state_dict(), buffer)
                buffer.seek(0)
                
                # Store in session state for download
                st.session_state.model_buffer = buffer.getvalue()
                st.session_state.download_ready = True
                st.session_state.download_filename = f"{model_name}.pth"
                
                st.success("Model ready for download!")
                
            except Exception as e:
                st.error(f"Error preparing download: {str(e)}")
        else:
            st.warning("Please enter a model name")
    
    # Download button (only show if model is ready)
    if hasattr(st.session_state, 'download_ready') and st.session_state.download_ready:
        st.download_button(
            label="📥 Download Model",
            data=st.session_state.model_buffer,
            file_name=st.session_state.download_filename,
            mime="application/octet-stream",
            use_container_width=True,
            help="Download the trained model file"
        )

